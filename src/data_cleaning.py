import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from scipy import stats

cwd = Path.cwd()

def load_data(data_file_name): ## Load data directly 
    data_path = cwd / f"data/{data_file_name}"
    data = pd.read_csv(data_path)
    print(data_path)
    return data



def quick_analysis(df): ## Get a quick overview of the dataset 
 print("Data Types:")
 print(df.dtypes,end="\n\n")
 print("Rows and Columns:")
 print(df.shape,end="\n\n")
 print("Column Names:")
 print(df.columns.tolist(), end="\n\n")
 print("Null Values:" )
 print(df.apply(lambda x: sum(x.isnull()) / len(df)), end="\n\n")



def normalize(data , features_to_normalize): ## normalize any list of features in any dataframe
    scaler = preprocessing.MinMaxScaler()
    
    data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])
    
    return data
   
def create_bins(data, column_name, bins, labels, new_column_name): ## Use when you want to divide a column into further bins/categorical columns
    data[new_column_name] = pd.cut(
        data[column_name],
        bins=bins,
        labels=labels
    )
    
    return data

def data_cleaning(df):
    df = pd.read_csv("nse_data.csv")
    print(df.head())
    print(df.info())

#Fix Column Names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    print(df.columns)

#Convert Date Column
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])

#Handle Missing Values
    print(df.isnull().sum())
    df = df.dropna()
#For financial data:
    df['delivery_percent'] = df['delivery_percent'].fillna(0)

#Remove Duplicate Rows
    df = df.drop_duplicates()

#Fix Data Types
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

#then check again
    df.info()

#Remove Impossible Values
    df = df[df['volume'] > 0]
    df = df[df['close'] > 0]
    df = df[df['delivery_percent'] <= 100]

#Detect Extreme Outliers
    df['z_volume'] = stats.zscore(df['volume'])
    df = df[df['z_volume'].abs() < 5]

#from scipy import stats

    df['z_volume'] = stats.zscore(df['volume'])
    df = df[df['z_volume'].abs() < 5]


#statistical information about numerical column
    df.describe()

#Create Clean Final Dataset
    df.to_csv("cleaned_nse_data.csv", index=False)

    return df
