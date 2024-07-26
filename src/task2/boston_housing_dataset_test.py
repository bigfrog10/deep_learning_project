import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
# import keras


# CRIM = crime rate
# ZN = proportion of residential land zoned for lots over 25000 square ft
# INDUS = proportion of non-retail business acres per town
# CHAS = charles river dummy variable
# NOX = nitric oxide concentration
# RM = average number of rooms
# AGE = proportion of occupied units before 1940
# DIS = weighted distance to 5 boston employment centers
# RAD = accessibility to radial highways
# TAX = full value property tax rate per 10k
# PTRATIO = pupil to teacher ratio by town
# LSTAT = % of lower status of population
# B = proportion of black neighbors -> housing values
# MEDV = median value of owner occupied homes in 1k dollars

# does not work anymore as of 1.2
# from sklearn.datasets import load_boston
# boston = load_boston()

df = pd.read_csv('housing_data.csv')
# print(df.head())
# print(df.info())
# print(df.describe().T)
#
# print(df.shape)
# print(df.corr)

# Plotting the heatmap of correlation between features
# plt.figure(figsize=(20, 20))
# sns.heatmap(df.corr(), cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')
# plt.show()
#
# df_features = df.copy()

# dataset = np.loadtxt('housing_data.csv', skiprows=1)
# [all rows, all columns except last one] *need iloc
# data_x = df.iloc[:, :-1]
# # [all rows, last column only]
# data_y = df.iloc[:, -1]
# train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.20)
# print(train_x)
X, y = df.values[:, :-1], df.values[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# scaler = StandardScaler()
# scaler = MinMaxScaler()
# train_x_s = scaler.fit_transform(train_x)
# test_x_s = scaler.transform(test_x)
# mean = train_x.mean(axis=0)
# std = train_x.std(axis=0)
# train_x -= mean
# train_x /= std
#
# test_x -= mean
# test_x /= std

# plotting box plots to see how the data looks, and if there are any outliers. like our assumptions say, there are.
# total_items = len(df.columns)
# items_per_row = 3
# total_rows = math.ceil(total_items / items_per_row)
#
# fig = make_subplots(rows=total_rows, cols=items_per_row)
#
# cur_row = 1
# cur_col = 1
#
# for index, column in enumerate(df.columns):
#     fig.add_trace(go.Box(y=df[column], name=column), row=cur_row, col=cur_col)
#
#     if cur_col % items_per_row == 0:
#         cur_col = 1
#         cur_row = cur_row + 1
#     else:
#         cur_col = cur_col + 1
#
#
# fig.update_layout(height=1000, width=550,  showlegend=False)
# fig.show()

# outlier, min data value, lower quartile value, median value, upper quartile value, max data value, outliers
# scatter plot below to find correlations between the target variable + other factors

# total_items = len(df.columns)
# items_per_row = 3
# total_rows = math.ceil(total_items / items_per_row)
#
# fig = make_subplots(rows=total_rows, cols=items_per_row, subplot_titles=df.columns)
#
# cur_row = 1
# cur_col = 1
#
# for index, column in enumerate(df.columns):
#     fig.add_trace(go.Scattergl(x=df[column],
#                                y=df['MEDV'],
#                                mode="markers",
#                                marker=dict(size=3)),
#                   row=cur_row,
#                   col=cur_col)
#
#     # intercept = np.poly1d(np.polyfit(df[column], df['MEDV'], 1))(np.unique(df[column]))
#
#     fig.add_trace(go.Scatter(x=np.unique(df[column]),
#                              y=intercept,
#                              line=dict(color='red', width=1)),
#                   row=cur_row,
#                   col=cur_col)
#
#     if cur_col % items_per_row == 0:
#         cur_col = 1
#         cur_row = cur_row + 1
#     else:
#         cur_col = cur_col + 1
#
#
# fig.update_layout(height=1000, width=550, showlegend=False)
# fig.show()

# model.add(layers.Input((13, )))
# model.add(layers.Dense(10, activation='relu', kernel_initializer='he_normal', name='dense_1', input_shape=(13,)))
# model.add(layers.Dense(8, activation='relu', kernel_initializer='he_normal', name='dense_2'))
# model.add(layers.Dense(1, activation='linear', name='dense_output'))

n_features = X_train.shape[1]

model = models.Sequential()
model.add(layers.Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(layers.Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(layers.Dense(1))
# compile the model
model.compile(optimizer='adam', loss='mse')

# batch_size = 128
# epochs = 100

# mean squared error, adaptive moment estimation, mean absolute error,
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.summary()

# history = model.fit(train_x, train_y, epochs=epochs, validation_split=0.1)
history = model.fit(X_train, y_train, epochs=150, batch_size=32)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

# print(history.history.keys())

# mse_nn, mae_nn = model.evaluate(test_x, test_y)
#
# print('Mean squared error on test data: ', mse_nn)
# print('Mean absolute error on test data: ', mae_nn)
