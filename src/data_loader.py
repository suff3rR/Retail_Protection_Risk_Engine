import pandas as pd
from pathlib import Path

# Columns that every valid raw stock CSV must have
REQUIRED_COLUMNS = {"symbol", "date", "close"}


def load_all_data(data_folder="data"):
    path = Path(data_folder)
    if not path.is_absolute():
        # Interpret relative data paths as relative to the repo root (one level above /src),
        path = Path(__file__).resolve().parent.parent / path

    all_files = list(path.glob("*.csv"))

    if not all_files:
        raise FileNotFoundError(f"No CSV files found in '{data_folder}/'")

    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        cols = set(df.columns.str.lower())

        if not REQUIRED_COLUMNS.issubset(cols):
            print(f"  Skipping '{file.name}' — not a stock data file")
            continue

        df_list.append(df)

    if not df_list:
        raise ValueError(
            "No valid stock CSV files found in data/.\n"
            "Each stock CSV must have at minimum: symbol, date, close columns."
        )

    return pd.concat(df_list, ignore_index=True)