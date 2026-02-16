import os
import pandas as pd
import numpy as np

try:
    from polygon import RESTClient
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'polygon'. Install with `pip install polygon-api-client`."
    ) from exc

# Initialize Client
api_key = os.getenv("POLYGON_API_KEY")
if not api_key:
    raise SystemExit("Set POLYGON_API_KEY before running this script.")
client = RESTClient(api_key)

def get_spread_acceleration(ticker1, ticker2, limit=100):
    # 1. Fetch 5-minute OHLCV data
    # multiplier=5, timespan="minute"
    aggs1 = client.get_aggs(ticker1, 5, "minute", "2025-01-01", "2025-12-31", limit=limit)
    aggs2 = client.get_aggs(ticker2, 5, "minute", "2025-01-01", "2025-12-31", limit=limit)
    
    df1 = pd.DataFrame(aggs1)
    df2 = pd.DataFrame(aggs2)
    
    # 2. Align timestamps and calculate the spread
    df = pd.merge(df1[['timestamp', 'close']], df2[['timestamp', 'close']], on='timestamp', suffixes=('_m1', '_m2'))
    df['spread'] = df['close_m1'] - df['close_m2']
    
    # 3. Calculate Velocity (1st Derivative)
    df['velocity'] = df['spread'].diff()
    
    # 4. Calculate Acceleration (2nd Derivative)
    df['acceleration'] = df['velocity'].diff()
    
    # 5. Calculate Z-Score of Acceleration (Reversion Signal)
    window = 20 # 20 periods of 5 mins = ~1.5 hours of trend
    df['accel_mean'] = df['acceleration'].rolling(window).mean()
    df['accel_std'] = df['acceleration'].rolling(window).std()
    df['z_score'] = (df['acceleration'] - df['accel_mean']) / df['accel_std']
    
    return df.tail(1)

# Example: CL March vs CL April
# result = get_spread_acceleration("O:CL2025H", "O:CL2025J")
