import pandas as pd
from pathlib import Path
from datetime import date, datetime
from tui import TUI, TUI_HEADING

def get_filepath(ticker: str) -> str:
    return "data/stocks_%s.csv" % ticker.lower()

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

class Database:
    def __init__(self):
        self.stocks: list[Stock] = []
        self.tickers: list[str] = []
        self.stocks_dict: dict[str, Stock] = {}
    
    def get(self, ticker: str) -> Stock:
        return self.stocks_dict[ticker]
    
    def clear(self):
        self.stocks.clear()
    
    def load_index(self):
        df = pd.read_csv("data/index.csv")
        for row in df.itertuples(index=False):
            ticker = getattr(row, 'Ticker')
            self.tickers.append(ticker)
    
    def update(self):
        self.clear()
        for ticker in self.tickers:
            if exists(get_filepath(ticker)):
                self.load_stock_ticker(ticker)
            
    def load_stock(self, df: pd.DataFrame, ticker: str):
        df = df.rename(columns={'Close/Last': 'Close'})

        stock_obj = Stock(name=ticker)
        for row in reversed(list(df.itertuples(index=False))): # Data is stored in reverse order for some reason, so reverse the frames
            frame = StockFrame()
            frame.name =   ticker
            frame.date =   getattr(row, 'Date')
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