import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime
import os

# Configurable variables
spreadsheet_path = "sheet.csv"  # Path to your CSV file
symbol = "GBPUSD"                     # Symbol to trade
lot = 0.1                              # Lot size

# Initialize connection to MetaTrader 5
if not mt5.initialize():
    print("Failed to initialize MetaTrader5, error code:", mt5.last_error())
    quit()

def read_spreadsheet(file_path):
    try:
        # Read the CSV file
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=None)  # No header in CSV, read as plain rows
            return df
        else:
            print(f"CSV file {file_path} not found.")
            return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def open_trade(action_type, symbol, lot):
    # Prepare the trade request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": action_type,
        "deviation": 20,
        "magic": 234000,
        "comment": "Hourly trade bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Send the trade request
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to send order: {result.retcode} at {datetime.now()}")
    else:
        print(f"Order sent successfully at {datetime.now()}: {result}")

def main():
    last_trade_time = None
    while True:
        current_time = datetime.now()
        if last_trade_time is None or (current_time - last_trade_time).seconds >= 3600:
            # Read CSV data
            df = read_spreadsheet(spreadsheet_path)
            if df is not None and not df.empty:
                # Get the value from the last row, first column
                value = df.iloc[-1, 0]
                # Get current market price
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    print(f"Failed to get tick data for {symbol}")
                else:
                    current_price = tick.ask if value > 0 else tick.bid
                    if value < current_price:  # If the value is lower than the current price, sell
                        open_trade(mt5.ORDER_TYPE_SELL, symbol, lot)
                    elif value > current_price:  # If the value is higher than the current price, buy
                        open_trade(mt5.ORDER_TYPE_BUY, symbol, lot)
                    else:
                        print(f"No action for value {value} compared to current price {current_price}")
            last_trade_time = current_time
        time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Bot interrupted by user.")

# Shutdown MT5 connection
mt5.shutdown()
