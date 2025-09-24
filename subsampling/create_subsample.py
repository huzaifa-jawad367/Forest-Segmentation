import pandas as pd
from pathlib import Path


def load_sorted_training_data(csv_path: str = "training_data_analysis.csv"):
    """Return the training data sorted by forest_percentage descending."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    df_sorted = df.sort_values("forest_percentage", ascending=False)

    # Ensure numeric types before computing cumulative sums
    for column in ("forest_pixels", "background_pixels", "total_pixels"):
        df_sorted[column] = pd.to_numeric(df_sorted[column], errors="coerce")

    df_sorted["cumulative_forest_pixels"] = df_sorted["forest_pixels"].cumsum()
    df_sorted["cumulative_background_pixels"] = df_sorted["background_pixels"].cumsum()
    df_sorted["cumulative_total_pixels"] = df_sorted["total_pixels"].cumsum()

    return df_sorted


if __name__ == "__main__":
    sorted_df = load_sorted_training_data()
    with pd.option_context(
        "display.max_columns", None,
        "display.width", None,
        "display.max_colwidth", None,
    ):
        print(sorted_df.to_string(index=False))

    output_path = Path(__file__).resolve().parent / "training_data_sorted.csv"
    sorted_df.to_csv(output_path, index=False)
    print(f"Saved sorted data to {output_path}")
