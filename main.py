# main.py
print("Script started execution")
import os
import sys
import pickle # Used for loading the saved model and scaler
import time # Used for sleeping between execution loops
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import * # Import trading constants (SIDE_BUY, SIDE_SELL, etc.)
import pandas as pd
import numpy as np
# ... other imports ...

# Global variables for the model and scaler
MODEL = None
SCALER = None

def load_ml_assets():
    """Loads the trained model and scaler objects from the disk."""
    global MODEL, SCALER
    try:
        with open('trading_bot_model.pkl', 'rb') as file:
            MODEL = pickle.load(file)
        with open('feature_scaler.pkl', 'rb') as file:
            SCALER = pickle.load(file)
        print("✅ ML Model and Scaler loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: ML asset not found. Did you download it from Colab? {e}")
        sys.exit(1)

def calculate_indicators(df):
    """
    Applies the necessary technical indicators to the DataFrame.
    MUST match the calculations used during Colab training.
    """
    WINDOW_FAST = 20
    WINDOW_SLOW = 50

    # A. SMA
    df['SMA_20'] = df['Close'].rolling(window=WINDOW_FAST).mean()
    df['SMA_50'] = df['Close'].rolling(window=WINDOW_SLOW).mean()

    # B. RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=14 - 1, min_periods=14).mean()
    avg_loss = loss.ewm(com=14 - 1, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # C. MACD
    EMA_12 = df['Close'].ewm(span=12, adjust=False).mean()
    EMA_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = EMA_12 - EMA_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # We need enough data (at least 50 points for the SMA_50) to avoid NaN rows
    return df.dropna()

def execute_trade(client, symbol, price):
    """
    Fetches the necessary historical data, makes a prediction, and executes a trade.
    """
    global MODEL, SCALER
    
    # 1. Fetch enough recent data to calculate all indicators (e.g., last 100 hours)
    # The 'Close' time of the last candle is NOW.
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_1HOUR,
        start_str='100 hours ago UTC' 
    )
    
    # Create DataFrame (similar to Phase 4)
    DATA_COLUMNS = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                    'Close Time', 'Quote Asset Volume', 'Number of Trades', 
                    'Taker Buy Base Vol', 'Taker Buy Quote Vol', 'Ignore']
    df = pd.DataFrame(klines, columns=DATA_COLUMNS)
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
    df.set_index('Open Time', inplace=True)
    
    # 2. Calculate Indicators
    df_with_indicators = calculate_indicators(df.copy())
    
    # Get the last row (the current, most complete candle)
    latest_features = df_with_indicators.iloc[-1]
    
    # Remove prediction targets and irrelevant columns that were dropped during training
    # Ensure this list EXACTLY matches what was dropped during training!
    irrelevant_cols = ['Close', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 
                       'Taker Buy Base Vol', 'Taker Buy Quote Vol', 'Ignore']
    
    # Filter the feature series to only include the columns used in training (indicators)
    feature_cols = [col for col in latest_features.index if col not in irrelevant_cols]
    
    # Prepare the final feature set for prediction
    X_live = latest_features[feature_cols].values.reshape(1, -1) 
    
    # 3. Scale the Data
    X_live_scaled = SCALER.transform(X_live)
    
    # 4. Predict
    prediction = MODEL.predict(X_live_scaled)[0]
    prediction_proba = MODEL.predict_proba(X_live_scaled)[0]
    
    print(f"Prediction: {'UP (1)' if prediction == 1 else 'DOWN/FLAT (0)'}, Proba: {prediction_proba}")
    
    
    # 5. Trading Logic (Simple Example)
    if prediction == 1 and prediction_proba[1] > 0.8: # Only trade if highly confident
        print(f"*** STRONG BUY SIGNAL! Placing order... ***")
        try:
            # You need a function to calculate the quantity based on your balance!
            order = client.create_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=0.001 # CHANGE THIS TO A CALCULATED VALUE!
            )
            print("ORDER PLACED:", order)
        except Exception as e:
            print(f"❌ ORDER FAILED: {e}")
    else:
        print("No strong signal or predicted down/flat. Holding position.")

def main():
    """Main function to run the trading bot."""
    print("---------------------------------------")
    print("Automated Trading Bot Initializing...")

    # --- 1. Load Keys (This section MUST remain at the top) ---
    # Load environment variables (from .env)
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    secret_key = os.getenv("BINANCE_SECRET_KEY")
    use_testnet = os.getenv("TESTNET", "0") == "1"

    if not api_key or not secret_key:
        # THE ERROR IS THROWN HERE if the .env file is not found/read.
        print("❌ ERROR: API keys not found! Ensure .env file is present.")
        sys.exit(1) # Program STOPS here if keys are missing.

    print(f"✅ Keys loaded successfully. Using Testnet: {use_testnet}")

    # --- 2. Initialize Binance Client ---
    try:
        client = Client(api_key, secret_key, testnet=use_testnet) 
        print("✅ Binance Client initialized.")
    except Exception as e:
        print(f"❌ ERROR initializing Binance Client: {e}")
        sys.exit(1)
        
    # --- 3. Load Trained Model and Scaler ---
    load_ml_assets()

    # --- 4. Start the Live Execution Loop ---
    SYMBOL = 'BTCUSDT'
    TRADE_INTERVAL_SECONDS = 3600 # Check every 1 hour (3600 seconds)

    print("\n🚀 Starting Automated Trading Loop...")
    print("---------------------------------------")

    while True:
        try:
            # Check for signal and execute trade
            # The execute_trade function handles fetching the *latest* data needed.
            execute_trade(client, SYMBOL, 0)
            
            print(f"\nSleeping for {TRADE_INTERVAL_SECONDS} seconds...")
            time.sleep(TRADE_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("\n🛑 Bot stopped manually.")
            break
        except Exception as e:
            print(f"🔥 UNEXPECTED ERROR in main loop: {e}. Restarting cycle...")
            time.sleep(30)
if __name__ == "__main__":
    main()