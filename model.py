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

BATCH_SIZE: int = 64
WINDOW_SIZE: int = 40

MODEL_FILENAME: str = "data/model.keras"

# Log returns:         5 values
# Intraday momentum:   3 values
# Rolling returns:     3 values
# Volatility features: 3 values
# Volume features:     3 values
# Moving averages:     4 values
STOCK_FEATURES: int = 21

class StockModel:
    def __init__(self):
        self._model: Sequential = Sequential([
            # Preprocessing
            Input((WINDOW_SIZE, STOCK_FEATURES)), # Input size

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
    
    def predict(self, stock: Stock, t: int=-1, apply_noise=False) -> np.float32:
        start = max(1, len(stock.frames) - WINDOW_SIZE - 51 - t)
        sequence = np.array([
            self.preprocess_frame(stock, i)
            for i in range(start, start + WINDOW_SIZE)
        ], dtype=np.float32)

        if apply_noise:
            rng = np.random.default_rng()
            bias = rng.uniform(low=-0.01, high=0.01, size=sequence.shape)
            sequence = sequence + bias
        
        sequence = np.expand_dims(sequence, axis=0)
        pred = self._model.predict(sequence, verbose=0)[0] # type: ignore
        return pred[0]
    
    def predict_range(self, stock: Stock, ts: list, apply_noise=False):
        sequences = []
        for t in ts:
            start = max(1, len(stock.frames) - WINDOW_SIZE - 50 - t)
            sequence = np.array([
                self.preprocess_frame(stock, i)
                for i in range(start, start + WINDOW_SIZE)
            ], dtype=np.float32)

            if apply_noise:
                rng = np.random.default_rng()
                bias = rng.uniform(low=-0.01, high=0.01, size=sequence.shape)
                sequence = sequence + bias
            
            sequences.append(sequence)
        
        sequences_arr = np.array(sequences, dtype=np.float32)
        preds = self._model.predict(sequences_arr, batch_size=512, verbose=0)[0] # type: ignore
        return preds
    
    def forecast(self, stock, t=-1) -> (tuple[int, float] | None):
        std_threshold = 0.1

        # Generate a bunch of (slightly) randomly modified guesses
        ts = [t for _ in range(50)]
        preds = self.predict_range(stock, ts, True)

        if np.abs(np.std(preds)) < std_threshold:
            mean = np.mean(preds)
            return (int(mean), float(mean))
        else:
            return None

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
        dataset = tf.data.Dataset.from_generator(
            self._sample_generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(WINDOW_SIZE, STOCK_FEATURES),
                    dtype=tf.float32
                ),
                tf.TensorSpec(
                    shape=(1,),
                    dtype=tf.float32
                )
            )
        )

        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        print("\t- Start training")
        history = self._model.fit(
            dataset,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            #validation_split=0.1,
            shuffle=True,
            verbose=1 # type: ignore
        )

        if save:
            self.save(filename)
        
        return history

    def _sample_generator(self):
        for stock in self.stocks:
            if len(stock.frames) < WINDOW_SIZE + 50:
                continue

            stock_features = np.array([
                self.preprocess_frame(stock, i)
                for i in range(50, len(stock.frames))
            ], dtype=np.float32)

            for i in range(WINDOW_SIZE, len(stock_features)):
                sequence = stock_features[i-WINDOW_SIZE:i]

                target = float(
                    stock.frames[i + 50].close >
                    stock.frames[i + 49].close
                )

                yield sequence, np.array([target], dtype=np.float32)
    
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
                returns = stock.log_returns(j)
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

        # Avoid division by zero
        self.std[self.std == 0] = 1.0
    
    def preprocess_frame(self, stock: Stock, t: int) -> np.ndarray:
        values = stock.log_returns(t)
        momentum = stock.intraday_momentum(t)
        returns = stock.rolling_returns(t)
        volatility = stock.volatility_features(t)
        volume = stock.volume_features(t)
        mov_avgs = stock.moving_averages(t)

        out = np.empty(STOCK_FEATURES, dtype=np.float32) # 21 Stock features

        out[:5] = (values - self.mean) / self.std
        out[5:8] = momentum
        out[8:11] = returns
        out[11:14] = volatility
        out[14:17] = volume
        out[17:21] = mov_avgs

        return out

    def load_stocks(self, data: Database):
        print("=== Loading Stocks into Model ===")
        self.stocks = data.stocks
        self.preprocess_stocks()