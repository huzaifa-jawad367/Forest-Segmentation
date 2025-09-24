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


def add_cumulative_ratio_columns(sorted_csv_path: Path, output_csv_path: Path | None = None):
    """Read sorted CSV and add cumulative ratio columns before saving a new file."""
    df = pd.read_csv(sorted_csv_path)
    for column in (
        "cumulative_forest_pixels",
        "cumulative_background_pixels",
        "cumulative_total_pixels",
    ):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    valid_totals = df["cumulative_total_pixels"].replace({0: pd.NA})
    df["cumm_forest_percent"] = df["cumulative_forest_pixels"] / valid_totals
    df["cumm_background_percent"] = df["cumulative_background_pixels"] / valid_totals

    if output_csv_path is None:
        output_csv_path = sorted_csv_path.with_name("training_data_sorted_with_cumm_percent.csv")

    df.to_csv(output_csv_path, index=False)
    return output_csv_path


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

    ratio_output_path = add_cumulative_ratio_columns(output_path)
    print(f"Saved cumulative ratio data to {ratio_output_path}")
