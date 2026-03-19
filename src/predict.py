import joblib
import numpy as np

def predict_risk(df, model_path):

    features = ['open','low','close','volume','manipulation_score']
    X = df[features]

    model = joblib.load(model_path)

    anomaly_score = model.decision_function(X)

    ml_risk_score = 100 * (
        1 - (anomaly_score - anomaly_score.min()) /
        (anomaly_score.max() - anomaly_score.min())
    )

    return ml_risk_score