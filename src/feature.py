import pandas as pd
import numpy as np
import yfinance as yf

def rename_columns(df):
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        "symbol" : "symbol",
        "date" : "date",
        "close" : "close",
        "total_volume" : "volume",
        "delivery_volume" : "deliverable_volume",
        "marketCap" : "marketCap"
    })

    return df

def get_yfinance_ticker(df):

    symbol = df['symbol'].iloc[0]

    return symbol + ".NS"
    
def get_shares_outstanding(symbol):
    
    ticker = yf.Ticker(symbol)
    
    info = ticker.info
    
    return info.get("sharesOutstanding")



def add_market_cap(df, symbol):

    shares = get_shares_outstanding(symbol)

    df["shares_outstanding"] = shares
    df["marketCap"] = df["close"] * shares

    return df

def add_manipulation_features(df):
    df = df.sort_values(["symbol", "date"]).copy()
    df["pct_change"] = df.groupby("symbol")["close"].pct_change()
 
    df["vol_20_avg"] = (
        df.groupby("symbol")["volume"]
        .transform(lambda x: x.rolling(20).mean())
    )
    df["volume_multiplier"] = df["volume"] / df["vol_20_avg"]
    df["is_volume_spike"] = (df["volume_multiplier"] > 5).astype(int)
 
    # ── 3. Upper circuit streak ──────────────────────────────────────────────
    # Approximation: >= 4.9% daily move = hitting upper circuit band
    df["hit_upper_circuit"] = (df["pct_change"] >= 0.049).astype(int)
    df["upper_circuit_streak"] = (
        df.groupby("symbol")["hit_upper_circuit"]
        .apply(lambda x: x * (x.groupby((x == 0).cumsum()).cumcount() + 1))
        .reset_index(level=0, drop=True)
    )
 
    # ── 4. Delivery % divergence ─────────────────────────────────────────────
    # Price rising but delivery % falling = speculative/manipulative buying
    df["delivery_pct"] = (df["deliverable_volume"] / df["volume"]) * 100
    df["delivery_5d_avg"] = (
        df.groupby("symbol")["delivery_pct"]
        .transform(lambda x: x.rolling(5).mean())
    )
    df["suspicious_delivery"] = (
        (df["pct_change"] > 0.05) &
        (df["delivery_pct"] < df["delivery_5d_avg"])
    ).astype(int)
 
    # ── 5. Price–volume correlation ──────────────────────────────────────────
    # Negative correlation = price moving without genuine buying pressure
    df["price_volume_corr"] = (
        df.groupby("symbol")
        .apply(lambda x: x["close"].rolling(10).corr(x["volume"]))
        .reset_index(level=0, drop=True)
    )
    df["negative_corr_flag"] = (df["price_volume_corr"] < -0.3).astype(int)
 
    # ── 6. NEW: Volume–price divergence ─────────────────────────────────────
    # High volume + flat or falling price = distribution phase
    # (insiders selling into retail volume, classic pump exit signal)
    df["volume_price_divergence"] = (
        (df["volume_multiplier"] > 2) &
        (df["pct_change"] <= 0)
    ).astype(int)
 
    # ── 7. NEW: Price z-score ────────────────────────────────────────────────
    # Flags statistically extreme price moves vs the stock's own 20-day history
    # Avoids the absolute ₹50 cutoff that penalised both stocks permanently
    df["close_20_mean"] = (
        df.groupby("symbol")["close"]
        .transform(lambda x: x.rolling(20).mean())
    )
    df["close_20_std"] = (
        df.groupby("symbol")["close"]
        .transform(lambda x: x.rolling(20).std())
    )
    df["price_zscore"] = (
        (df["close"] - df["close_20_mean"]) /
        df["close_20_std"].replace(0, np.nan)
    )
    df["extreme_price_move"] = (df["price_zscore"].abs() > 2).astype(int)
 
    # ── 8. Penny stock flag (reference only — NOT in manipulation_score) ─────
    # Kept for human review but excluded from scoring.
    # Both stocks are almost always under ₹50, making this a constant, not a signal.
    df["is_penny_stock"] = (
        (df["close"] < 50) |
        (df["marketCap"] < 500e7)
    ).astype(int)
 
    # ── 9. Manipulation score (max possible = 6) ─────────────────────────────
    df["manipulation_score"] = (
        df["is_volume_spike"] +                              # 1 pt
        (df["upper_circuit_streak"] >= 3).astype(int) +     # 1 pt
        df["suspicious_delivery"] +                          # 1 pt
        df["negative_corr_flag"] +                           # 1 pt
        df["volume_price_divergence"] +                      # 1 pt
        df["extreme_price_move"]                             # 1 pt
    )
 
    # ── 10. is_manipulated flag ──────────────────────────────────────────────
    # Threshold lowered from 3 → 2.
    # Old threshold: 2/730 rows flagged (useless for training).
    # New threshold: gives the model enough positive examples to learn from.
    df["is_manipulated"] = (df["manipulation_score"] >= 2).astype(int)
 
    return df



def add_pump_dump_features(df):

    df = df.sort_values(['symbol', 'date'])
    df['daily_return'] = df.groupby('symbol')['close'].pct_change()
    df['price_change_5d'] = df.groupby('symbol')['close'].pct_change(5)
    df['avg_volume_5d'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(5).mean())
    df['volume_spike'] = df['volume'] / df['avg_volume_5d']
    df['volatility_5d'] = df.groupby('symbol')['daily_return'].transform(lambda x: x.rolling(5).std())

    def zscore(x):
        return (x - x.mean()) / x.std()

    df['volume_zscore'] = df.groupby('symbol')['volume'].transform(zscore)
    df['return_zscore'] = df.groupby('symbol')['daily_return'].transform(zscore)

    df['pump_strength'] = df['price_change_5d'] * df['volume_spike']

    df['dump_strength'] = df['daily_return'].rolling(3).sum()

    df['price_acceleration'] = df.groupby('symbol')['daily_return'].diff()

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    return df

