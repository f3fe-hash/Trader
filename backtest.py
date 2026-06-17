import numpy as np

from stock import Database, Stock
from model import StockModel, MODEL_FILENAME, WINDOW_SIZE

def forecast(model: StockModel, stock: Stock):
    tup = model.forecast(stock, -1)
    if tup is not None:
        pred = tup[0]
        acc = tup[1]
        print("Model predicted %s would go %s with %.2f%% certainty" % (
            stock.name,
            "up" if pred == 1 else "down",
            acc
        ))

def backtest():
    # Load in the data and model
    database = Database(do_load_index=True)
    model = StockModel()
    model.load_stocks(database)
    model.load(MODEL_FILENAME)

    print("Start")
    correct = 0
    num = 0
    for stock in model.stocks:
            length = len(stock.frames)
            if length < WINDOW_SIZE + 50:
                continue

            capital = 10000

            for i, k in enumerate(range(WINDOW_SIZE, length - 49)):
                next = stock.frames[k + 50].close
                curr = stock.frames[k + 49].close
                pred = model.forecast(stock, i)

                if not pred:
                    continue

                if pred:
                    capital *= (1.0 + (next - curr) / curr)

                if pred == (next > curr):
                    correct += 1
                num += 1
            
            print("Completed stock %s\t| Profit: %.2f" % (stock.name, capital - 10000))
    
    print("Correct: %d\t| Total: %d" % (correct, num))

if __name__ == '__main__':
    backtest()