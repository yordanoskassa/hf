import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Example model
from sklearn.metrics import mean_squared_error


import schedule
import time
from datetime import datetime

def predict_and_save():
    # Download EUR/GBP data for the past 6 months with hourly intervals
    eur_gbp_data = yf.download("EURGBP=X", interval="1h", period="6mo")

    # Download GBP/JPY data for the past 6 months with hourly intervals
    gbp_jpy_data = yf.download("GBPJPY=X", interval="1h", period="6mo")



    gbp_usd_data = yf.download("GBPUSD=X", interval="1h", period="6mo")

    gbp_usd_data['GBP/USD_2hr'] = gbp_usd_data['Close'].shift(+1)

    gbp_jpy_data['GBP/USD_2hr'] = gbp_usd_data['GBP/USD_2hr']



    #RSI
    def calculate_rsi(data, window=14):
        delta = data['Close'].diff()  # Calculate the difference in 'Close' prices
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  # Calculate rolling mean of gains
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  # Calculate rolling mean of losses
        
        # To avoid division by zero, replace 0 losses with a small value
        loss = loss.replace(0, 1e-10)
        
        rs = gain / loss  # Calculate the relative strength (RS)
        rsi = 100 - (100 / (1 + rs))  # Compute RSI
        
        return rsi

    # Apply the function to your dataset
    gbp_jpy_data['GBPJPY_RSI'] = calculate_rsi(gbp_jpy_data)

    gbp_jpy_data['GBPUSD_RSI'] = calculate_rsi(gbp_usd_data)



    #ATR

    def calculate_atr(data, window=14):
        # Calculate the True Range (TR)
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close = (data['Low'] - data['Close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate the Average True Range (ATR)
        atr = true_range.rolling(window=window).mean()
        
        return atr

    # Apply the ATR function to the GBP/JPY dataset
    gbp_jpy_data['GBPJPY_ATR'] = calculate_atr(gbp_jpy_data)


    def calculate_sma(data, window=14):
        """
        Calculate Simple Moving Average (SMA).
        """
        sma = data['Close'].rolling(window=window).mean()  # Calculate the rolling mean
        return sma

    def calculate_ema(data, window=14):
        """
        Calculate Exponential Moving Average (EMA).
        """
        ema = data['Close'].ewm(span=window, adjust=False).mean()  # Calculate the exponentially weighted mean
        return ema

    # Add SMA and EMA columns to your dataframe
    gbp_jpy_data['GBPJPY_SMA_14'] = calculate_sma(gbp_jpy_data, window=14)  # 14-period SMA
    gbp_jpy_data['GBPJPY_EMA_14'] = calculate_ema(gbp_jpy_data, window=14)  # 14-period EMA

    eur_gbp_data['EURGBP_RSI'] = calculate_rsi(eur_gbp_data)
    eur_gbp_data['EURGBP_ATR'] = calculate_atr(eur_gbp_data)
    eur_gbp_data['EURGBP_SMA'] = calculate_sma(eur_gbp_data)
    eur_gbp_data['EURGBP_EMA'] = calculate_ema(eur_gbp_data)

    # Add EUR/GBP 'Close', RSI, ATR, SMA, EMA to the GBP/JPY dataset
    gbp_jpy_data['EURGBP_Close'] = eur_gbp_data['Close']
    gbp_jpy_data['EURGBP_RSI'] = eur_gbp_data['EURGBP_RSI']
    gbp_jpy_data['EURGBP_ATR'] = eur_gbp_data['EURGBP_ATR']
    gbp_jpy_data['EURGBP_SMA'] = eur_gbp_data['EURGBP_SMA']
    gbp_jpy_data['EURGBP_EMA'] = eur_gbp_data['EURGBP_EMA']


    gbp_usd_data['GBPUSD_SMA'] = calculate_sma(gbp_usd_data)
    gbp_usd_data['GBPUSD_EMA'] = calculate_ema(gbp_usd_data)

    #gbp_jpy_data['GBPUSD_SMA'] = gbp_usd_data['GBPUSD_SMA']
    #gbp_jpy_data['GBPUSD_EMA'] = gbp_usd_data['GBPUSD_EMA']


    gbp_jpy_data["GBP/JPY_Close"] = gbp_jpy_data["Close"]

    gbp_jpy_data = gbp_jpy_data.drop(columns=['Volume', 'Adj Close'])
    gbp_jpy_data = gbp_jpy_data.dropna()

    gbp_jpy_data_log_scaled = gbp_jpy_data.apply(lambda x: np.log1p(x) if np.issubdtype(x.dtype, np.number) else x)





    X = gbp_jpy_data.drop(columns=['GBP/USD_2hr'])  # Features (all columns except the target)
    y = gbp_jpy_data['GBP/USD_2hr']  # Target column

    # Step 2: Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Train the model (Example: Random Forest Regressor)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)  # Train the model on the training set

    # Step 4: Evaluate the model
    y_pred = model.predict(X_test)  # Make predictions on the test set
    mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error

    print(f"Mean Squared Error on the test set: {mse}")

    # Optional: View some predictions vs actual values
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(comparison_df.head())


    last_row = X.tail(1)  # Get the last row of features (without the target)

    # Make a prediction for the last row
    last_row_prediction = model.predict(last_row)



    # Output the prediction
    print(X.tail(1))

    print(f'last_row_prediction: {last_row_prediction}')



    with open("sheet.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        value = last_row_prediction[0]
        value = round(value, 5)
        lst = []
        lst.append(value)
        # Round to 5 decimal places
        # Write the value as a new row
        writer.writerow([i for i in lst])


schedule.every().hour.do(predict_and_save)

# Keep the script running to maintain the schedule
while True:
    current_time = datetime.now()
    # Check if it's the start of the hour (minute == 0)
    if current_time.minute == 0:
        predict_and_save()
        time.sleep(60)  # Wait a minute to avoid running multiple times within the same hour
    time.sleep(30)  # Check every 30 seconds
