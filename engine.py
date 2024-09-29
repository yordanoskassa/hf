import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


eurusd = yf.Ticker("EURUSD=X")

hist = eurusd.history(period="6mo")

print(hist)