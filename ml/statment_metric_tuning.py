from sklearn.neighbors import KNeighborsRegressor
import pandas
import numpy
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score

dt = datasets.load_boston()
y = dt.target
x = dt.data
x = scale(x)
gen = KFold(shuffle=True, n_splits=5, random_state=42)
print(x)
scores = list()
for i in numpy.linspace(1, 10, 200):
    reg = KNeighborsRegressor(n_neighbors=5, weights='distance', p=i)
    tmp = cross_val_score(gen, x, y, cv=reg, scoring='neg_mean_squared_error')
    tmp = numpy.array(tmp).mean()
    scores.append(tmp)
print(tmp)
