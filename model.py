import os
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Masking, Dropout, Conv1D, Input
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.preprocessing.sequence import pad_sequences

from stock import StockFrame, Database, Stock

BATCH_SIZE = 32
WINDOW_SIZE = 100

# Log returns:         5 values
# Intraday momentum:   3 values
# Rolling returns:     3 values
# Volatility features: 3 values
# Volume features:     3 values
# Moving averages:     4 values
STOCK_FEATURES = 21

class StockModel:
    def __init__(self):
        self._model: Sequential = Sequential([
            # Preprocessing
            Input((WINDOW_SIZE, STOCK_FEATURES)), # Input size (unknown b/c LSTM)

            # Actual model
            Conv1D(32, kernel_size=3, activation='relu'),
            LSTM(64, activation='tanh'),
            Dropout(0.2),
            Dense(32, activation='tanh'),
            Dropout(0.2),
            Dense(16, activation='tanh'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        self.stocks: list[Stock] = []
    
    def predict(self, stock: Stock):
        last = stock.frames[-1].close
        start = max(1, len(stock.frames) - WINDOW_SIZE)
        sequence = np.array([
            self.preprocess_frame(stock, i)
            for i in range(start, len(stock.frames))
        ], dtype=np.float32)
        
        sequence = np.expand_dims(sequence, axis=0)
        pred = self._model.predict(sequence, verbose=0)[0] # type: ignore
        return pred[0]
    
    def train(self, epochs, load=True, save=True, filename=""):
        print("=== Training the Model")
        if load:
            self.load(filename)
        
        print("\t- Compiling the model")
        lr_schedule = ExponentialDecay(
            1e-3, #0.001
            100000,
            0.9,
            staircase=False
        )
        
        self._model.compile(
            optimizer=Adam(lr_schedule), # type: ignore
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print("\t- Processing dataset")
        X = []
        y = []
        for stock in self.stocks:
            if len(stock.frames) < 2:
                continue

            stock_features = [
                self.preprocess_frame(stock, i)
                for i in range(50, len(stock.frames))
            ]

            # Create training samples
            # Start at 50 because some functions calculate datta for 50 days
            for i in range(WINDOW_SIZE + 50, len(stock.frames)):
                sequence = stock_features[i-WINDOW_SIZE:i]

                target_frame = stock.frames[i]
                prev_frame = stock.frames[i-1]

                close_diff = target_frame.close - prev_frame.close
                target = np.zeros(shape=(1), dtype=np.float32)
                target[0] = float(close_diff > 0)

                X.append(sequence)
                y.append(target)

        if not X:
            raise ValueError("No training samples generated")

        X = pad_sequences(
            X,
            padding='post',
            dtype='float32',
            value=0.0
        )

        y = np.array(y, dtype=np.float32)

        print("\t- Start training")
        history = self._model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            shuffle=True,
            verbose=1 # type: ignore
        )

        if save:
            self.save(filename)
        
        return history
    
    def save(self, filename):
        self._model.save(filename)
    
    def load(self, filename):
        if Path(filename).is_file():
            self._model = load_model(filename) # type: ignore
        else:
            print("%s doesn't exist." % filename)
            return
    
    def preprocess_stocks(self):
        print("=== Preprocessing the Dataset ===")
        opens = []
        closes = []
        lows = []
        highs = []
        volumes = []

        print("\t- Calculating log returns")
        for i in range(1, len(self.stocks)):
            stock = self.stocks[i]

            for j in range(1, len(stock.frames)):
                returns = stock.log_returns(i)
                opens.append(returns[0])
                closes.append(returns[1])
                lows.append(returns[2])
                highs.append(returns[3])
                volumes.append(returns[4])

        print("\t- Calculating mean of log returns")
        self.mean = np.array([
            np.mean(opens),
            np.mean(closes),
            np.mean(lows),
            np.mean(highs),
            np.mean(volumes)
        ], dtype=np.float32)

        print("\t- Calculating standard deviation of log returns")
        self.std = np.array([
            np.std(opens),
            np.std(closes),
            np.std(lows),
            np.std(highs),
            np.std(volumes)
        ], dtype=np.float32)

        #returns = []

        #for stock in self.stocks:
        #    for i in range(1, len(stock.frames)):
        #        r = np.log(stock.frames[i].close) - np.log(stock.frames[i - 1].close)
        #        returns.append(r)

        # Avoid division by zero
        self.std[self.std == 0] = 1.0

        #if self.return_std == 0:
        #    self.return_std = 1.0
    
    def preprocess_frame(self, stock: Stock, t: int) -> np.ndarray:
        values = stock.log_returns(t)             # 5 values
        momentum = stock.intraday_momentum(t)     # 3 values
        returns = stock.rolling_returns(t)        # 3 values
        volatility = stock.volatility_features(t) # 3 values
        volume = stock.volume_features(t)         # 3 values
        mov_avgs = stock.moving_averages(t)       # 4 values
        return np.concatenate([(values - self.mean) / self.std, momentum, returns, volatility, volume, mov_avgs])

    def load_stocks(self, data: Database):
        print("=== Loading Stocks into Model ===")
        self.stocks = data.stocks
        self.preprocess_stocks()