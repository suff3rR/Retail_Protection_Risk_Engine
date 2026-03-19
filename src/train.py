import joblib
import os
from pathlib import Path
from sklearn.ensemble import IsolationForest


# These are the only columns the model should learn from.
# Raw OHLCV (open, close, volume etc.) are excluded — IsolationForest
# would treat normal price levels as anomalies, which is meaningless.
# Only engineered signals that actually encode manipulation behaviour go in.
MODEL_FEATURES = [
    "volume_multiplier",        # how unusual today's volume is vs 20d avg
    "upper_circuit_streak",     # consecutive days hitting price ceiling
    "suspicious_delivery",      # price up but delivery % falling
    "negative_corr_flag",       # price moving without genuine buying
    "volume_price_divergence",  # high volume + flat/falling price (distribution)
    "extreme_price_move",       # statistically abnormal price move (z-score)
    "manipulation_score",       # composite rule score (0–6)
]


def train_model(df, model_path, force_retrain=True):
    """
    Trains or loads an IsolationForest model.
    Set force_retrain=True to ignore any saved model and retrain from scratch.
    """

    if os.path.exists(model_path) and not force_retrain:
        print(f"Loading existing model from {model_path}")
        return joblib.load(model_path)

    print("Training new model...")

    # Validate all expected features exist
    missing = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"Missing features in dataframe: {missing}\n"
            f"Make sure add_manipulation_features() ran before train_model()."
        )

    X = df[MODEL_FEATURES].fillna(0)

    model = IsolationForest(
        n_estimators=200,       # more trees = more stable scores
        contamination=0.10,     # ~10% — matches observed 7.8% fraud rate with buffer
        random_state=42,
    )

    model.fit(X)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved → {model_path}")

    return model