TRAIN_TICKERS = [
    # US large-caps
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM",
    "V", "WMT", "DIS", "NFLX", "KO", "PEP", "XOM", "PFE",
    # Indian large-caps (NSE)
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS", "HINDUNILVR.NS",
]

WATCHLIST = [
    "AAPL", "MSFT", "TSLA", "NVDA", "AMZN",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS",
]

HISTORY_PERIOD = "3y"      # how much history to pull for training
HORIZON_DAYS = 5           # look this many trading days ahead for the label
BUY_THRESHOLD = 0.02       # future 5-day return > +2%  -> Buy
SELL_THRESHOLD = -0.02     # future 5-day return < -2%  -> Sell
