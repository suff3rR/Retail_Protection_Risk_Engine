import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,IsolationForest
from sklearn.metrics import mean_absolute_error
import yfinance_data_download
from feature import add_manipulation_features,rename_columns
import sys
import joblib

sys.path.append(str(Path().resolve() / "src"))

def IsolationForest_function(test_data):

    test_features = ['open','low','close','volume','manipulation_score'] ## Add feature Variables
    y = test_data.high
    X = test_data[test_features] 
    train_X , val_X , train_y , val_y = train_test_split(X,y,random_state=1)
    test_model = IsolationForest(contamination=0.05)
    test_model.fit(train_X)

    test_predictions = test_model.predict(val_X)
    test_anomaly_score = test_model.decision_function(val_X)
    ml_risk_score = 100 * (1 - (test_anomaly_score - test_anomaly_score.min()) /
                        (test_anomaly_score.max() - test_anomaly_score.min()))
 
    val_X = val_X.copy()

    val_X['ml_risk_score'] = ml_risk_score

  
    val_X['rule_risk_score'] = (
        val_X['manipulation_score'] /
        val_X['manipulation_score'].max()
    ) * 100


    val_X['final_risk_score'] = (
        0.6 * val_X['ml_risk_score'] +
        0.4 * val_X['rule_risk_score']
    )

    def risk_label(score):
        if score >= 80:
            return "High Risk"
        elif score >= 60:
            return "Moderate Risk"
        elif score >= 30:
            return "Low Risk"
        else:
            return "Normal"

    val_X['risk_category'] = val_X['final_risk_score'].apply(risk_label)
    return val_X


if __name__ == "__main__":
    cwd = Path.cwd()
    dataset_path = cwd / "data/cleaned_nse_data.csv"
   
    df = pd.read_csv(dataset_path)
    df = rename_columns(df=df)
    df['marketCap'] = 1e9
    df = add_manipulation_features(df=df)
   
    results = IsolationForest_function(test_data=df)
    top_risky = results.sort_values("final_risk_score", ascending=False)
    print("Top 10 Most Suspicious Events")
    print(top_risky.head(10))