from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv

#Initializing API
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / '.kaggle/kaggle.json/.env'
load_dotenv(dotenv_path=env_path)

# Fetching Credentials
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
# Initializing Kaggle API
os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
os.environ["KAGGLE_KEY"] = KAGGLE_KEY

print("USERNAME:", os.getenv("KAGGLE_USERNAME"))
print("KEY:", os.getenv("KAGGLE_KEY"))
