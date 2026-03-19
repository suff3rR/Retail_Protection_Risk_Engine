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
 
    df["hit_upper_circuit"] = (df["pct_change"] >= 0.049).astype(int)
    df["upper_circuit_streak"] = (
        df.groupby("symbol")["hit_upper_circuit"]
        .apply(lambda x: x * (x.groupby((x == 0).cumsum()).cumcount() + 1))
        .reset_index(level=0, drop=True)
    )
 
    df["delivery_pct"] = (df["deliverable_volume"] / df["volume"]) * 100
    df["delivery_5d_avg"] = (
        df.groupby("symbol")["delivery_pct"]
        .transform(lambda x: x.rolling(5).mean())
    )
    df["suspicious_delivery"] = (
        (df["pct_change"] > 0.05) &
        (df["delivery_pct"] < df["delivery_5d_avg"])
    ).astype(int)
 
    df["price_volume_corr"] = (
        df.groupby("symbol")
        .apply(lambda x: x["close"].rolling(10).corr(x["volume"]))
        .reset_index(level=0, drop=True)
    )
    df["negative_corr_flag"] = (df["price_volume_corr"] < -0.3).astype(int)

    df["volume_price_divergence"] = (
        (df["volume_multiplier"] > 2) &
        (df["pct_change"] <= 0)
    ).astype(int)
 
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
 
    df["is_penny_stock"] = (
        (df["close"] < 50) |
        (df["marketCap"] < 500e7)
    ).astype(int)
 
    df["manipulation_score"] = (
        df["is_volume_spike"] +                              
        (df["upper_circuit_streak"] >= 3).astype(int) +     
        df["suspicious_delivery"] +                          
        df["negative_corr_flag"] +                          
        df["volume_price_divergence"] +                      
        df["extreme_price_move"]                             
    )

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

