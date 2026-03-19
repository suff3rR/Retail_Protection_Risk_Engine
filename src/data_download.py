from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import config

def download_dataset():
    api = KaggleApi()
    print("authentication...")
    api.authenticate()
    print("authentication successfull")
    download_path = Path("data/")
    download_path.mkdir(exist_ok=True)

    api.dataset_download_file(
    "borismarjanovic/price-volume-data-for-all-us-stocks-etfs",
    path=download_path,
    file_name="Data/ETFs/agg.us.txt"
    )

if __name__ == "__main__":
    download_dataset()


