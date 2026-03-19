def calculate_final_risk(df, ml_risk_score):

    df = df.copy()
    df['ml_risk_score'] = ml_risk_score

    # Normalize ML score to 0–100
    score_min = df['ml_risk_score'].min()
    score_max = df['ml_risk_score'].max()
    if score_max == score_min:
        df['ml_risk_score_norm'] = 50.0
    else:
        df['ml_risk_score_norm'] = 100 * (
            1 - (df['ml_risk_score'] - score_min) / (score_max - score_min)
        )

    # Normalize rule score to 0–100
    # Guard against divide-by-zero when manipulation_score is all 0s
    rule_max = df['manipulation_score'].max()
    if rule_max == 0 or rule_max != rule_max:  # 0 or NaN
        df['rule_risk_score'] = 0.0
    else:
        df['rule_risk_score'] = (df['manipulation_score'] / rule_max) * 100

    # Hybrid: 60% ML signal + 40% rule signal
    df['final_risk_score'] = (
        0.6 * df['ml_risk_score_norm'] +
        0.4 * df['rule_risk_score']
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

    df['risk_category'] = df['final_risk_score'].apply(risk_label)

    return df