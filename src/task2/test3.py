from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
import numpy as np
import pandas as pd

df = pd.read_csv('boston_housing_prices.csv')
total_items = len(df.columns)
items_per_row = 3
total_rows = math.ceil(total_items / items_per_row)

fig = make_subplots(rows=total_rows, cols=items_per_row, subplot_titles=df.columns)

cur_row = 1
cur_col = 1

for index, column in enumerate(df.columns):
    fig.add_trace(go.Scattergl(x=df[column],
                               y=df['MEDV'],
                               mode="markers",
                               marker=dict(size=3)),
                  row=cur_row,
                  col=cur_col)

    intercept = np.poly1d(np.polyfit(df[column], df['MEDV'], 1))(np.unique(df[column]))

    fig.add_trace(go.Scatter(x=np.unique(df[column]),
                             y=intercept,
                             line=dict(color='red', width=1)),
                  row=cur_row,
                  col=cur_col)

    if cur_col % items_per_row == 0:
        cur_col = 1
        cur_row = cur_row + 1
    else:
        cur_col = cur_col + 1


fig.update_layout(height=1000, width=550, showlegend=False)
fig.show()