import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


class LSTMModel:
    def __init__(self, window_size=12, epochs=100, batch_size=16):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.model = None
        self.last_window = None

    def _create_sequences(self, series):
        X, y = [], []
        for i in range(len(series) - self.window_size):
            X.append(series[i:i + self.window_size])
            y.append(series[i + self.window_size])
        return np.array(X), np.array(y)

    def fit(self, y: pd.Series):
        y = y.dropna()
        if len(y) <= self.window_size:
            raise ValueError(f"Series too short for window size {self.window_size}")

        scaled_y = self.scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        X, y_seq = self._create_sequences(scaled_y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.window_size, 1)),
            Dense(1)
        ])
        self.model.compile(optimizer=Adam(), loss='mse')
        self.model.fit(X, y_seq, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        self.last_window = scaled_y[-self.window_size:]

    def predict(self, steps: int = 1) -> pd.Series:
        if self.last_window is None or self.model is None:
            raise RuntimeError('Model must be fit before predicting.')

        window = self.last_window.copy()
        preds = []

        for _ in range(steps):
            X_input = window.reshape((1, self.window_size, 1))
            pred = self.model.predict(X_input, verbose=0)[0][0]
            preds.append(pred)
            window = np.append(window[1:], pred)

        preds = self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        return pd.Series(preds)


class MLPModel:
    def __init__(self, window_size=12, epochs=100, batch_size=16):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.model = None
        self.last_window = None

    def _create_sequences(self, series):
        X, y = [], []
        for i in range(len(series) - self.window_size):
            X.append(series[i:i + self.window_size])
            y.append(series[i + self.window_size])
        return np.array(X), np.array(y)

    def fit(self, y: pd.Series):
        y = y.dropna()
        if len(y) <= self.window_size:
            raise ValueError(f"Series too short for window size {self.window_size}")

        scaled_y = self.scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        X, y_seq = self._create_sequences(scaled_y)

        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(self.window_size,)),
            Dense(1)
        ])
        self.model.compile(optimizer=Adam(), loss='mse')
        self.model.fit(X, y_seq, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        self.last_window = scaled_y[-self.window_size:]

    def predict(self, steps: int = 1) -> pd.Series:
        if self.last_window is None or self.model is None:
            raise RuntimeError('Model must be fit before predicting.')

        window = self.last_window.copy()
        preds = []

        for _ in range(steps):
            X_input = window.reshape((1, self.window_size))
            pred = self.model.predict(X_input, verbose=0)[0][0]
            preds.append(pred)
            window = np.append(window[1:], pred)

        preds = self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        return pd.Series(preds)