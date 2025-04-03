import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class LSTMModel:
    def __init__(
        self,
        window_size=12,
        epochs=100,
        batch_size=16,
        hidden_units=64,
        activation='relu',
    ):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.activation = activation
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        y = y.dropna()

        if len(X) != len(y):
            raise ValueError('X and y must have the same length.')

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(
            y.values.reshape(-1, 1)
        ).flatten()

        X_seq = X_scaled.reshape(
            (X_scaled.shape[0], 1, X_scaled.shape[1])
        )  # (samples, time_step=1, features)

        self.model = Sequential(
            [
                LSTM(
                    self.hidden_units,
                    activation=self.activation,
                    input_shape=(1, X_seq.shape[2]),
                ),
                Dense(1),
            ]
        )
        self.model.compile(optimizer=Adam(), loss='mse')
        self.model.fit(
            X_seq,
            y_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )

    def predict(self, X_future: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError('Model must be fit before predicting.')

        X_scaled = self.scaler_X.transform(X_future)
        X_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        preds = self.model.predict(X_seq, verbose=0).flatten()
        return pd.Series(
            self.scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten(),
            index=X_future.index,
        )


class MLPModel:
    def __init__(
        self,
        window_size=12,
        epochs=100,
        batch_size=16,
        hidden_units=64,
        activation='relu',
    ):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.activation = activation
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        y = y.dropna()

        if len(X) != len(y):
            raise ValueError('X and y must have the same length.')

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(
            y.values.reshape(-1, 1)
        ).flatten()

        self.model = Sequential(
            [
                Dense(
                    self.hidden_units,
                    activation=self.activation,
                    input_shape=(X_scaled.shape[1],),
                ),
                Dense(1),
            ]
        )
        self.model.compile(optimizer=Adam(), loss='mse')
        self.model.fit(
            X_scaled,
            y_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )

    def predict(self, X_future: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError('Model must be fit before predicting.')

        X_scaled = self.scaler_X.transform(X_future)
        preds = self.model.predict(X_scaled, verbose=0).flatten()
        return pd.Series(
            self.scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten(),
            index=X_future.index,
        )
