import yfinance as yf

def load_data(stock_index="AAPL"):
    return yf.Ticker(stock_index).history(period="15y")