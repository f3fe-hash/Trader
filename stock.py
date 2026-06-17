import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime
from tui import TUI, TUI_HEADING

def get_filepath(ticker: str) -> str:
    return "data/stocks/%s.csv" % ticker.lower()

def exists(filename: str):
    return Path(filename).is_file()

def load_price(price: str) -> float:
    if price[0] == '$':
        return float(price[1:])
    else:
        return float(price)

class StockFrame:
    def __init__(self):
        self.name: str
        self.date: date
        self.open: float = 0.00
        self.close: float = 0.00
        self.low: float = 0.00
        self.high: float = 0.00
        self.volume: int = 0

class Stock:
    def __init__(self, name=None):
        self.name: str | None = name
        self.frames: list[StockFrame] = []
    
    def append_frame(self, frame: StockFrame):
        self.frames.append(frame)
    
    def log_returns(self, t: int):
        if t < 1:
            raise ValueError("Need at least 1 prior frames")
        
        frame = self.frames[t]
        prev = self.frames[t - 1]

        volume_change = 0
        if (prev.volume == 0) or (frame.volume == 0):
            volume_change = 5
        else:
            volume_change = np.log(frame.volume / prev.volume)
        
        return np.array([
            np.log(frame.open / prev.close),
            np.log(frame.close / prev.close),
            np.log(frame.low / prev.close),
            np.log(frame.high / prev.close),
            volume_change
        ], dtype=np.float32)

    def intraday_momentum(self, t: int):
        frame = self.frames[t]

        return np.array([
            (frame.close - frame.open) / frame.open, # Close-open ratio
            (frame.high - frame.low) / frame.close, # High-low range
            (frame.close - frame.low) / (frame.high - frame.low + 1e-8) # Close position
        ], dtype=np.float32)
    
    def rolling_returns(self, t: int):
        if t < 14:
            raise ValueError("Need at least 14 prior frames")
        
        frame = self.frames[t]
        return np.array([
            (frame.close / self.frames[t - 3].close) - 1,
            (frame.close / self.frames[t - 7].close) - 1,
            (frame.close / self.frames[t - 14].close) - 1,
        ], dtype=np.float32)
    
    def volatility_features(self, t: int):
        if t < 14:
            raise ValueError("Need at least 14 prior frames")
        
        rolling_std_5 = np.std([np.log(self.frames[i].close / self.frames[i - 1].close) for i in range(t - 4, t)])
        rolling_std_10 = np.std([np.log(self.frames[i].close / self.frames[i - 1].close) for i in range(t - 9, t)])
        atr = np.mean([self.frames[i].high / self.frames[i - 1].low for i in range(t - 13, t)]) # 14 day window
        return np.array([
            rolling_std_5,
            rolling_std_10,
            atr
        ], dtype=np.float32)
    
    def volume_features(self, t: int):
        if t < 10:
            raise ValueError("Need at least 10 prior frames")
        
        frame = self.frames[t]
        prev = self.frames[t - 1]

        volume_change = 0
        if (prev.volume == 0) or (frame.volume == 0):
            volume_change = np.log1p(frame.volume) - np.log1p(prev.volume)
        else:
            volume_change = np.log(frame.volume / prev.volume)
        
        volume_ma_ratio = frame.volume / (np.mean([frame.volume for frame in self.frames[t-10:t]]) + 1e-8)
        price_volume_trend = np.log(frame.close / prev.close) * volume_change
        return np.array([
            volume_change,
            volume_ma_ratio,
            price_volume_trend
        ], dtype=np.float32)
    
    def moving_averages(self, t: int):
        if t < 50:
            raise ValueError("Need at least 50 prior frames")
        
        frame = self.frames[t]

        def sma(period):
            if t < period - 1:
                return None

            closes = [f.close for f in self.frames[t-period+1:t+1]]
            return np.mean(closes)

        sma_5 = sma(5)
        sma_10 = sma(10)
        sma_20 = sma(20)
        sma_50 = sma(50)

        sma_5_ratio = frame.close / sma_5 if sma_5 else 1.0
        sma_10_ratio = frame.close / sma_10 if sma_10 else 1.0
        sma_20_ratio = frame.close / sma_20 if sma_20 else 1.0
        sma_50_ratio = frame.close / sma_50 if sma_50 else 1.0

        return np.array([
            sma_5_ratio - 1,
            sma_10_ratio - 1,
            sma_20_ratio - 1,
            sma_50_ratio - 1
        ], dtype=np.float32)

class Database:
    def __init__(self, do_load_index=False):
        self.stocks: list[Stock] = []
        self.tickers: list[str] = []
        self.stocks_dict: dict[str, Stock] = {}

        if do_load_index:
            self.load_index()
            self.update()
    
    def get(self, ticker: str) -> Stock:
        return self.stocks_dict[ticker]
    
    def clear(self):
        self.stocks.clear()
    
    def load_index(self):
        print("=== Loading Stock Index ===")
        df = pd.read_csv("data/index.csv")
        for row in df.itertuples(index=False):
            ticker = getattr(row, 'Ticker')
            self.tickers.append(ticker)
    
    # Update missing stock files by replacing them
    def update(self):
        print("=== Updating Stock Database ===")
        self.clear()
        for ticker in self.tickers:
            if exists(get_filepath(ticker)):
                self.load_stock_ticker(ticker)
            else:
                self.download_stock(ticker)
    
    # Forcefully update every stock file.
    def update_force(self):
        print("=== Updating Stock Database ===")
        self.clear()
        for ticker in self.tickers:
            self.download_stock(ticker)

    def download_stock(self, ticker: str):
        ticker = ticker.replace(".", "-")
        file_path = get_filepath(ticker)

        df = yf.download(
            ticker,
            period="max",
            interval="1d",
            auto_adjust=False,
            group_by="column",   # important
            progress=False
        )

        assert df is not None

        if df.empty:
            print(f"[WARN] No data for {ticker}")
            return

        # FIX: flatten MultiIndex columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        # Ensure proper column names
        df = df.rename(columns={"index": "Date"})

        # Keep only what you need
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        Path("data").mkdir(exist_ok=True, parents=True)
        df.to_csv(file_path, index=False)

        self.load_stock(df, ticker)

        print(f"[INFO] Downloaded {ticker}")
            
    def load_stock(self, df: pd.DataFrame, ticker: str):
        df = df.rename(columns={'Close/Last': 'Close'})

        stock_obj = Stock(name=ticker)
        for row in df.itertuples(index=False):
            frame = StockFrame()
            frame.name =   ticker
            frame.date =   row[0]
            frame.volume = getattr(row, 'Volume')
            frame.open =   load_price(str(getattr(row, 'Open')))
            frame.close =  load_price(str(getattr(row, 'Close')))
            frame.low =    load_price(str(getattr(row, 'Low')))
            frame.high =   load_price(str(getattr(row, 'High')))

            stock_obj.append_frame(frame)

        self.stocks.append(stock_obj)
        self.stocks_dict[ticker] = stock_obj

    def load_stock_ticker(self, ticker: str):
        df = pd.read_csv(get_filepath(ticker), parse_dates=['Date'])
        self.load_stock(df, ticker)

    def print(self, screen: TUI):
        for stock in self.stocks:
            name = stock.frames[0].name
            screen.print("=== %s ===" % name, TUI_HEADING)
            for i, frame in enumerate(stock.frames):
                screen.print("Frame %2d  | Date: %s  | Open: $%.2f  | Volume: %d  | Close: $%.2f  | Low: $%.2f  | High: $%.2f" % (i,
                    frame.date.strftime("%Y-%m-%d"), frame.open, frame.volume, frame.close, frame.low, frame.high))
            screen.skip_rows(1)
        screen.update()