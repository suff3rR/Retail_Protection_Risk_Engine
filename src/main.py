import pandas as pd
from pathlib import Path
from data_loader import load_all_data
from feature import rename_columns, add_manipulation_features, add_market_cap, get_yfinance_ticker
from train import train_model, MODEL_FEATURES
from risk_scoring import calculate_final_risk
from ground_truth import build_ground_truth
from evaluate import evaluate_model
from datetime import datetime
import os

GROUND_TRUTH_PATH = "data/labeled_ground_truth.csv"
timestamp = datetime.now().strftime("%Y_%m_%d")
MODEL_PATH = f"models/isolation_forest_{timestamp}.pkl"


if __name__ == "__main__":

    df = load_all_data()
    df = rename_columns(df)
    symbol = get_yfinance_ticker(df)
    df = add_market_cap(df, symbol)
    df = add_manipulation_features(df)

    model = train_model(df, MODEL_PATH, force_retrain=False)

    X = df[MODEL_FEATURES].fillna(0)
    ml_scores = model.decision_function(X)


    results = calculate_final_risk(df, ml_scores)

    summary = (
        results.groupby("symbol")
        .agg(
            avg_risk_score=("final_risk_score", "mean"),
            max_risk_score=("final_risk_score", "max"),
            manipulated_days=("is_manipulated", "sum"),
            total_days=("is_manipulated", "count"),
        )
        .reset_index()
    )

    summary["manipulation_rate"] = (
        summary["manipulated_days"] / summary["total_days"] * 100
    ).round(1)

    summary["avg_risk_score"] = summary["avg_risk_score"].round(2)
    summary["max_risk_score"] = summary["max_risk_score"].round(2)

    def risk_label(score):
        if score >= 80:
            return "High Risk"
        elif score >= 60:
            return "Moderate Risk"
        elif score >= 30:
            return "Low Risk"
        else:
            return "Normal"

    summary["risk_category"] = summary["avg_risk_score"].apply(risk_label)
    summary["is_manipulated"] = summary["manipulated_days"] > 0

    display_cols = ["symbol", "is_manipulated", "avg_risk_score", "max_risk_score", "manipulation_rate", "risk_category"]

    print("\n── Stock Manipulation Summary ────────────────────────────────────")
    print(summary[display_cols].sort_values("avg_risk_score", ascending=False).to_string(index=False))
    print()

    print("\n── Building ground truth labels ──────────────────────────────────")
    build_ground_truth(data_folder="data", output_path=GROUND_TRUTH_PATH)

    evaluate_model(results, ground_truth_path=GROUND_TRUTH_PATH) 