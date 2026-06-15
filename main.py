from stock import Database
from model import StockModel

LOAD_MODEL: bool    = False
SAVE_MODEL: bool    = True
TRAIN_MODEL: bool   = True
TRAIN_EPOCHS: int   = 10
MODEL_FILENAME: str = "data/model.keras"

def forecast(ticker: str):
    global database
    global model
    stock = database.get(ticker)
    direction = model.predict(stock)

    print("Direction of ticker %s: %d" % (ticker, direction))

# Main function
def main():
    global database
    global model
    
    # Load in all of the stock data
    database = Database()
    database.load_index()
    database.update()

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