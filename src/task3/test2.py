import pandas as pd
import sklearn.linear_model

df = pd.read_csv('fd01-sample1.csv')

import seaborn as sns
sns.displot(df, x='Class')

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

# scaler = MinMaxScaler()
# train_x_s = scaler.fit_transform(X_train)
# test_x_s = scaler.transform(X_train)
x = preprocessing.normalize(df)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x, df.Class)
print(model.coef_)
y = model.predict(x)

# import seaborn as sns
# sns.catplot(x=x[:, 0], y=model.labels_)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
cm = confusion_matrix(df.iloc[:, 0], y)
print(cm)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', annot_kws={'size': 20})

plt.show()

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.labels_)
# disp.plot()
# plt.show()
