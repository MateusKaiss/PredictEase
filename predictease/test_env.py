import numpy as np
import pandas as pd
import torch
# import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import matplotlib

print("\n Verificando pacotes e versões...")

print(f" NumPy versão: {np.__version__}")
print(f" Pandas versão: {pd.__version__}")
print(f" Torch versão: {torch.__version__}")
print(f" XGBoost versão: {xgb.__version__}")
print(f" LightGBM versão: {lgb.__version__}")
print(f" Seaborn versão: {sns.__version__}")
print(f" Matplotlib versão: {matplotlib.__version__}")
print(f" Optuna versão: {optuna.__version__}")
print(f" pmdarima versão: {pm.__version__}")
print(f" Statsmodels versão: {ARIMA.__module__}")
print(f" Scikit-Learn versão: {LinearRegression.__module__}")
print(f" Prophet versão: {Prophet.__module__}")
# print(f" TensorFlow versão: {tf.__version__}")

print("\n Todos os pacotes foram importados e verificados com sucesso!")
