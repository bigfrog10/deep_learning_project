import tensorflow as tf
import pandas as pd
from numpy import sqrt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
from sklearn.metrics import r2_score
print(tf.__version__)

# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = pd.read_csv('boston_housing_prices.csv')

print(df.head())
print(df.info())
print(df.describe().T)

print(df.shape)
print(df.corr)

# Plotting the heatmap of correlation between features
plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')
plt.show()

# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# plotting box plots to see how the data looks, and if there are any outliers. like our assumptions say, there are.
total_items = len(df.columns)
items_per_row = 3
total_rows = math.ceil(total_items / items_per_row)

fig = make_subplots(rows=total_rows, cols=items_per_row)

cur_row = 1
cur_col = 1

for index, column in enumerate(df.columns):
    fig.add_trace(go.Box(y=df[column], name=column), row=cur_row, col=cur_col)

    if cur_col % items_per_row == 0:
        cur_col = 1
        cur_row = cur_row + 1
    else:
        cur_col = cur_col + 1


fig.update_layout(height=1000, width=550,  showlegend=False)
fig.show()

# determine the number of input features
n_features = X_train.shape[1]

# define model
model = Sequential()
model.add(layers.Input((13, )))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(layers.Dropout(0.2))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))

# compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

# fit the model
history = model.fit(X_train, y_train, epochs=500, batch_size=256, validation_split=0.05)

# plot loss
fig = go.Figure()
fig.add_trace(go.Scattergl(y=history.history['loss'],
                           name='Train'))

fig.add_trace(go.Scattergl(y=history.history['val_loss'],
                           name='Test'))


fig.update_layout(height=500, width=700,
                  xaxis_title='Epoch',
                  yaxis_title='Loss')

fig.show()

# plot mean absolute error
fig = go.Figure()
fig.add_trace(go.Scattergl(y=history.history['mae'],
                           name='Train'))

fig.add_trace(go.Scattergl(y=history.history['val_mae'],
                           name='Test'))


fig.update_layout(height=500, width=700,
                  xaxis_title='Epoch',
                  yaxis_title='Mean Absolute Error')

fig.show()

# evaluate the model
error = model.evaluate(X_test, y_test, verbose=0)
print(dict(zip(model.metrics_names, error)))

# print('MSE: %.3f, RMSE: %.3f' % (error, sqrt(error)))
print("results:", error)

# make a prediction
row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
yhat = model.predict(X_test)
print(f'Predicted:{yhat}')

# r2score
r2 = r2_score(y_test, yhat)
print(r2)
