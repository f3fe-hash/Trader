import os
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Masking, Input
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.preprocessing.sequence import pad_sequences

from stock import StockFrame, Database, Stock

BATCH_SIZE = 32

# Open, Close, High, Low, Volume
STOCK_FEATURES = 5

class StockModel:
    def __init__(self):
        self._model: Sequential = Sequential([
            # Preprocessing
            Input((None, STOCK_FEATURES)), # Input size (unknown b/c LSTM)
            Masking(mask_value=0.0), # Skip the padding

            # Actual model
            LSTM(64, activation='tanh'),
            Dense(32, activation='tanh'),
            Dense(16, activation='tanh'),
            Dense(1, activation='linear')
        ])

        self.stocks: list[Stock] = []
    
    def predict(self, stock: Stock):
        last = stock.frames[-1].close
        WINDOW = 100
        start = max(1, len(stock.frames) - WINDOW)
        sequence = np.array([
            self.preprocess_frame(stock.frames[i], stock.frames[i - 1])
            for i in range(start, len(stock.frames))
        ], dtype=np.float32)
        
        sequence = np.expand_dims(sequence, axis=0)
        pred = self._model.predict(sequence, verbose="")[0][0]
        pred = pred * self.return_std + self.return_mean # Denormalize
        pred = np.exp(pred) * last # Un-log it
        return pred
    
    def train(self, epochs, load=True, save=True, filename=""):
        if load:
            self.load(filename)
        
        lr_schedule = ExponentialDecay(
            1e-3, #0.001
            100000,
            0.9,
            staircase=False
        )
        
        self._model.compile(
            optimizer=Adam(lr_schedule), # type: ignore
            loss='mse',
            metrics=['mse', 'mae']
        )

        X = []
        y = []
        WINDOW = 100
        for stock in self.stocks:
            if len(stock.frames) < 2:
                continue

            # Create training samples
            for i in range(WINDOW + 1, len(stock.frames)):
                sequence = [
                    self.preprocess_frame(stock.frames[j], stock.frames[j - 1])
                    for j in range(i-WINDOW,i)
                ]

                target_frame = stock.frames[i]
                prev_frame = stock.frames[i-1]

                target = np.log(target_frame.close / prev_frame.close)
                target = (target - self.return_mean) / self.return_std

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

        print(f"Training samples: {len(X)}")
        print(f"Input shape: {X.shape}")

        print("y mean:", np.mean(y))
        print("y std:", np.std(y))
        print("y min:", np.min(y))
        print("y max:", np.max(y))

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
        opens = []
        closes = []
        lows = []
        highs = []
        volumes = []

        for i, stock in enumerate(self.stocks):
            for j in range(1, len(stock.frames)):
                frame = stock.frames[j]
                prev = stock.frames[j - 1]

                if (prev.volume == 0) or (frame.volume == 0):
                    continue

                ref = prev.close
                opens.append  (np.log(frame.open   / ref))
                closes.append (np.log(frame.close  / ref))
                lows.append   (np.log(frame.low    / ref))
                highs.append  (np.log(frame.high   / ref))
                volumes.append(np.log(frame.volume / prev.volume))

        self.mean = np.array([
            np.mean(opens),
            np.mean(closes),
            np.mean(lows),
            np.mean(highs),
            np.mean(volumes)
        ], dtype=np.float32)

        self.std = np.array([
            np.std(opens),
            np.std(closes),
            np.std(lows),
            np.std(highs),
            np.std(volumes)
        ], dtype=np.float32)

        returns = []

        for stock in self.stocks:
            for i in range(1, len(stock.frames)):
                r = np.log(stock.frames[i].close) - np.log(stock.frames[i - 1].close)
                returns.append(r)

        self.return_mean = np.mean(returns)
        self.return_std = np.std(returns)
        print("return_mean: %.10f" % self.return_mean)
        print("return_std: %.10f" % self.return_std)

        # Avoid division by zero
        self.std[self.std == 0] = 1.0

        if self.return_std == 0:
            self.return_std = 1.0
    
    def preprocess_frame(self, frame, prev) -> np.ndarray:
        values = np.array([
            np.log(frame.open / prev.close),
            np.log(frame.close / prev.close),
            np.log(frame.low / prev.close),
            np.log(frame.high / prev.close),
            np.log(frame.volume / prev.volume)
        ], dtype=np.float32)

        return (values - self.mean) / self.std

    def load_stocks(self, data: Database):
        self.stocks = data.stocks
        self.preprocess_stocks()