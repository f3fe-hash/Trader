from stock import Database
from model import StockModel, MODEL_FILENAME

LOAD_MODEL: bool    = True
SAVE_MODEL: bool    = True
TRAIN_MODEL: bool   = True
TRAIN_EPOCHS: int   = 55

def forecast(ticker: str):
    global database
    global model
    stock = database.get(ticker)
    direction = model.forecast(stock)

    print("Direction of ticker %s: %d" % (ticker, direction))

# Main function
def main():
    global database
    global model
    
    # Load in all of the stock data
    database = Database(do_load_index=True)

    # Load the stock data into the model
    model = StockModel()
    model.load_stocks(database)
    
    if TRAIN_MODEL:
        model.train(TRAIN_EPOCHS, LOAD_MODEL, SAVE_MODEL, MODEL_FILENAME)
    else:
        if LOAD_MODEL:
            model.load(MODEL_FILENAME)

    # AAPL, AMZN, AVGO, GOOG, META, MSFT, NVDA, TSLA
    forecast("AAPL")
    forecast("AMZN")
    forecast("AVGO")
    forecast("GOOG")
    forecast("META")
    forecast("MSFT")
    forecast("NVDA")
    forecast("TSLA")

if __name__ == '__main__':
    main()