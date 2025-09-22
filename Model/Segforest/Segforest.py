import torch
import torch.nn as nn
from transformers import SegformerModel, SegformerConfig
from Model.Segforest.Mix_transformer import MixVisionTransformer, mit_b4
from Model.Segforest.MFF_blocks import MFFBlocks
from Model.Segforest.MSMD import MultiScaleMultiDecoder

class CrossChannelFusionAll(nn.Module):
    """
    Early cross-attention for an arbitrary number of channels C.
    For each query channel q, we attend to the concatenation of ALL other channels (KV).
    C == 1 ⇒ no-op (residual).
    Steps per forward:
      1) 1x1 conv to embed each channel to dim d (result: (B, C*d, H, W))
      2) reshape to token sequences per channel (B, T, d)
      3) for each q: MHA(q, cat(KV of all other channels), cat(...))
      4) concat [attn_out, residual_q] → linear → (B, T, d)
      5) merge all channels back via 1x1 conv and residual-add to input
    """
    def __init__(self,
                 in_chans: int,
                 dim: int = 64,
                 num_heads: int = 4,
                 attn_dropout: float = 0.1,
                 proj_dropout: float = 0.0):
        super().__init__()
        assert in_chans >= 1
        self.C = in_chans
        self.d = dim
        self.h = num_heads

        # Per-channel embedding to common hidden dim
        self.embed = nn.Conv2d(in_chans, in_chans * dim, kernel_size=1, bias=False)

        # Shared MHA used repeatedly (batch_first=True expects (B, T, d))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        # Per-query projector after concatenating [attn_out, residual_q]
        self.proj_q = nn.Linear(dim + dim, dim)

        # Merge C channel embeddings back to C planes
        self.merge = nn.Conv2d(in_chans * dim, in_chans, kernel_size=1, bias=False)

        self.dropout_attn = nn.Dropout(attn_dropout)
        self.dropout_proj = nn.Dropout(proj_dropout)
        self.norm = nn.LayerNorm(dim)

    def _split_channels(self, x_emb):
        # x_emb: (B, C*d, H, W) -> list length C of (B, d, H, W)
        B, CD, H, W = x_emb.shape
        d = self.d
        xs = [x_emb[:, i*d:(i+1)*d, :, :] for i in range(self.C)]
        return xs, H, W

    def _ensure_shapes_for(self, C_current, device, dtype):
        """If the module was constructed for a different C, rebuild lightweight layers."""
        if C_current != self.C:
            self.C = C_current
            self.embed = nn.Conv2d(self.C, self.C * self.d, kernel_size=1, bias=False).to(device=device, dtype=dtype)
            self.merge = nn.Conv2d(self.C * self.d, self.C, kernel_size=1, bias=False).to(device=device, dtype=dtype)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        self._ensure_shapes_for(C, x.device, x.dtype)

        # (B, C, H, W) -> (B, C*d, H, W)
        x_emb = self.embed(x)
        xs, H, W = self._split_channels(x_emb)  # list of C tensors (B, d, H, W)

        T = H * W
        seqs = [t.permute(0, 2, 3, 1).reshape(B, T, self.d) for t in xs]  # (B, T, d)
        seqs_norm = [self.norm(s) for s in seqs]

        # Trivial early exit when only one channel
        if C == 1:
            return x

        # Precompute per-channel sequences for concatenated KV
        # For each q, KV = concat of all channels except q along token dimension
        new_seqs = []
        for q in range(C):
            q_seq = seqs_norm[q]  # (B, T, d)
            kv_cat = torch.cat([seqs_norm[kv] for kv in range(C) if kv != q], dim=1)  # (B, (C-1)*T, d)

            attn_out, _ = self.attn(q_seq, kv_cat, kv_cat)  # (B, T, d)
            attn_out = self.dropout_attn(attn_out)

            # Concat with residual of original q, then project back to d
            mixed = torch.cat([attn_out, seqs[q]], dim=-1)  # (B, T, 2d)
            mixed = self.proj_q(mixed)                      # (B, T, d)
            mixed = self.dropout_proj(mixed)
            new_seqs.append(mixed)

        # Back to (B, C*d, H, W)
        new_maps = [s.reshape(B, H, W, self.d).permute(0, 3, 1, 2) for s in new_seqs]
        fused_feat = torch.cat(new_maps, dim=1)             # (B, C*d, H, W)

        # Project to C planes and residual add
        delta = self.merge(fused_feat)                      # (B, C, H, W)
        out = x + delta
        return out
        
class Segforest(nn.Module):
    def __init__(self, 
                 img_size=128, 
                 in_chans=3,  # Changed to 3 for RGB images
                 encoder_embed_dims=[64, 128, 320, 512],
                 mff_out_channels=[64, 128, 320],
                 decoder_inner_channels=64,
                 num_classes=2,  # Changed to 2 for binary segmentation
                 pretrained_model_name="nvidia/mit-b4",  # Pretrained Segformer model
                 freeze_encoder=False # Option to freeze encoder weights
                 # ===== NEW: simple toggles/params for all-to-all cross-channel fusion =====
                 use_cross_channel_fusion=True,
                 cc_dim=64,
                 cc_heads=4,
                 cc_attn_dropout=0.1,
                 cc_proj_dropout=0.0
                ):  
        super().__init__()
        
        # NEW: all-to-all cross-channel fusion (works for any in_chans >= 1)
        self.use_ccf = use_cross_channel_fusion
        if self.use_ccf:
            self.ccf = CrossChannelFusionAll(
                in_chans=in_chans,
                dim=cc_dim,
                num_heads=cc_heads,
                attn_dropout=cc_attn_dropout,
                proj_dropout=cc_proj_dropout
            )
        
        # Load pretrained Segformer encoder
        self.encoder = SegformerModel.from_pretrained(
            pretrained_model_name,
            ignore_mismatched_sizes=True  # Ignore size mismatches for different input channels
        )
        
        # Update the first layer if input channels don't match
        if in_chans != 3:
            # Replace the first conv layer to handle different input channels
            old_conv = self.encoder.patch_embeddings[0].proj
            new_conv = nn.Conv2d(
                in_chans, 
                old_conv.out_channels, 
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            
            # Initialize new conv layer
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            if new_conv.bias is not None:
                nn.init.constant_(new_conv.bias, 0)
            
            self.encoder.patch_embeddings[0].proj = new_conv
        
        # Freeze encoder weights if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        # MFF blocks: input channels are the sum of encoder outputs at each scale after concat
        # For MFFBlocks, in_channels_list = [sum of channels after concat for k=1,2,3]
        # Easch concat is 4 encoder outputs resized and concatenated, so sum of encoder_embed_dims
        mff_in_channels = [sum(encoder_embed_dims)] * 3
        self.mff_blocks = MFFBlocks(mff_in_channels, mff_out_channels)
        # Decoder: TB4 channels is encoder_embed_dims[3]
        self.decoder = MultiScaleMultiDecoder(
            mff_channels=mff_out_channels, 
            embed_dims=encoder_embed_dims,
            inner_channels=decoder_inner_channels,
            num_classes=num_classes
        )

    def forward(self, x):
        # ===== NEW: apply all-to-all cross-channel fusion BEFORE the encoder =====
        if self.use_ccf:
            x = self.ccf(x)  # (B, C, H, W) -> fused (B, C, H, W)
        # Get encoder outputs from pretrained Segformer
        encoder_outputs = self.encoder(x, output_hidden_states=True, return_dict=True)
        
        # Extract the hidden states (multi-scale features)
        # Segformer returns hidden_states as a tuple of 4 feature maps
        hidden_states = encoder_outputs.hidden_states  # [TB1, TB2, TB3, TB4]
        
        # Convert to the expected format for MFF blocks
        # Reshape from (B, N, C) to (B, C, H, W) format
        reshaped_outputs = []
        for i, hidden_state in enumerate(hidden_states):
            B, N, C = hidden_state.shape
            # Calculate H, W based on the stage (each stage has different resolution)
            H = W = int(N ** 0.5)  # Assuming square feature maps
            reshaped = hidden_state.reshape(B, H, W, C).permute(0, 3, 1, 2)
            reshaped_outputs.append(reshaped)
        
        mff_outputs = self.mff_blocks(reshaped_outputs)  # [MFF_1, MFF_2, MFF_3]
        decoder_outputs = self.decoder(mff_outputs, reshaped_outputs)  # [out1, out2, out3]
        
        if self.training:
            # Training mode: return list of 3 upsampled outputs
            upsampled_outputs = [
                torch.nn.functional.interpolate(decoder_outputs[0], scale_factor=4, mode='bilinear', align_corners=False),
                torch.nn.functional.interpolate(decoder_outputs[1], scale_factor=2, mode='bilinear', align_corners=False),
                torch.nn.functional.interpolate(decoder_outputs[2], scale_factor=2, mode='bilinear', align_corners=False)
            ]
            return upsampled_outputs
        else:
            # Evaluation mode: return only the highest resolution output with scale factor 4
            return [torch.nn.functional.interpolate(decoder_outputs[0], scale_factor=4, mode='bilinear', align_corners=False)]
