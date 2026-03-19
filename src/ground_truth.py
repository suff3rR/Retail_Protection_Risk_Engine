import pandas as pd
from pathlib import Path

def build_ground_truth(data_folder="data", output_path="data/labeled_ground_truth.csv"):
    """
    Extracts ground truth fraud labels from bulk_deal_flag column
    across all raw stock CSVs in the data folder.

    Saves a CSV with columns: symbol, date, is_fraud
    is_fraud = 1 if bulk_deal_flag == YES, else 0
    """
    path = Path(data_folder)
    output_file = Path(output_path).resolve()
    all_files = list(path.glob("*.csv"))

    if not all_files:
        raise FileNotFoundError(f"No CSV files found in '{data_folder}/'")

    df_list = []
    for file in all_files:

        if file.resolve() == output_file:
            continue

        df = pd.read_csv(file, parse_dates=["date"])

        if "bulk_deal_flag" not in df.columns:
            print(f"  Skipping '{file.name}' — no bulk_deal_flag column")
            continue

        if "symbol" not in df.columns or "date" not in df.columns:
            print(f"  Skipping '{file.name}' — missing symbol/date columns")
            continue

        df_list.append(df[["symbol", "date", "bulk_deal_flag"]])

    if not df_list:
        raise ValueError(
            "No valid stock CSV files with bulk_deal_flag found.\n"
            "Make sure your raw stock CSVs are in the data/ folder."
        )

    combined = pd.concat(df_list, ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "date"])

    combined["is_fraud"] = (combined["bulk_deal_flag"].str.upper() == "YES").astype(int)
    ground_truth = combined[["symbol", "date", "is_fraud"]]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ground_truth.to_csv(output_path, index=False)

    total = len(ground_truth)
    fraud_count = ground_truth["is_fraud"].sum()
    print(f"Ground truth saved → {output_path}")
    print(f"  Total rows : {total}")
    print(f"  Fraud days : {fraud_count} ({100 * fraud_count / total:.1f}%)")
    print(f"  Clean days : {total - fraud_count}")

    return ground_truth


if __name__ == "__main__":
    build_ground_truth()