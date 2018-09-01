import numpy as np
import scipy
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
import pandas as pd

data = pd.read_csv(r'C:\Users\olllk\Downloads\wine.data', header=None)

y = data[0]
x = data.loc[:, 1:]
print(x)
gen = KFold(shuffle=True, n_splits=5, random_state=42)


def func_accuracy(gen, x, y):
    scores = list()
    for i in range(1, 51):
        clf = KNeighborsClassifier(i)
        tmp = cross_val_score(clf, x, y, cv=gen, scoring='accuracy')
        m = np.array(tmp).mean()
        scores.append(m)
    return pd.DataFrame(scores, range(1, 51)).sort_values(by=[0], ascending=False)


accuracy = func_accuracy(gen, x, y)
print(accuracy.head(1))

x = scale(x)
accuracy = func_accuracy(gen, x, y)
print(accuracy.head(1))