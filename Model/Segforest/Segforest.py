import torch
import torch.nn as nn
from transformers import SegformerModel, SegformerConfig
from Model.Segforest.Mix_transformer import MixVisionTransformer, mit_b4
from Model.Segforest.MFF_blocks import MFFBlocks
from Model.Segforest.MSMD import MultiScaleMultiDecoder

class Segforest(nn.Module):
    def __init__(self, 
                 img_size=128, 
                 in_chans=3,  # Changed to 3 for RGB images
                 encoder_embed_dims=[64, 128, 320, 512],
                 mff_out_channels=[64, 128, 320],
                 decoder_inner_channels=64,
                 num_classes=2,  # Changed to 2 for binary segmentation
                 pretrained_model_name="nvidia/mit-b5",  # Pretrained Segformer model
                 freeze_encoder=False):  # Option to freeze encoder weights
        super().__init__()
        
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
        # Get encoder outputs from pretrained Segformer
        encoder_outputs = self.encoder(x, output_hidden_states=True, return_dict=True)
        
        # Extract the hidden states (multi-scale features)
        # Segformer returns hidden_states as a tuple of 4 feature maps
        hidden_states = encoder_outputs.hidden_states  # [TB1, TB2, TB3, TB4]
        
        # Convert to the expected format for MFF blocks
        # Segformer outputs are already in (B, C, H, W) format, no reshaping needed
        reshaped_outputs = []
        for i, hidden_state in enumerate(hidden_states):
            # Debug: print the actual shape
            # print(f"Hidden state {i} shape: {hidden_state.shape}")
            
            # Check if it's already in (B, C, H, W) format
            if len(hidden_state.shape) == 4:
                # Already in correct format
                reshaped_outputs.append(hidden_state)
            elif len(hidden_state.shape) == 3:
                # Need to reshape from (B, N, C) to (B, C, H, W)
                B, N, C = hidden_state.shape
                # Calculate H, W based on the stage (each stage has different resolution)
                H = W = int(N ** 0.5)  # Assuming square feature maps
                reshaped = hidden_state.reshape(B, H, W, C).permute(0, 3, 1, 2)
                reshaped_outputs.append(reshaped)
            else:
                raise ValueError(f"Unexpected hidden state shape: {hidden_state.shape}")
        
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