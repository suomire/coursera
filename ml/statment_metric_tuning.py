from sklearn.neighbors import KNeighborsRegressor
import pandas
import numpy
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score

# Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston().
# Результатом вызова данной функции является объект,
#  у которого признаки записаны в поле data, а целевой вектор — в поле target.

dt = datasets.load_boston()
y = dt.target
x = dt.data

# Приведите признаки в выборке к одному масштабу при помощи функции sklearn.preprocessing.scale.
x = scale(x)
# Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом, чтобы всего было протестировано
# 200 вариантов (используйте функцию numpy.linspace). Используйте KNeighborsRegressor
#  с n_neighbors=5 и weights='distance' — данный параметр добавляет в алгоритм веса,
#  зависящие от расстояния до ближайших соседей. В качестве метрики качества используйте
# среднеквадратичную ошибку (параметр scoring='mean_squared_error' у cross_val_score;
#  при использовании библиотеки scikit-learn версии 0.18.1 и выше необходимо указывать
# scoring='neg_mean_squared_error'). Качество оценивайте, как и в предыдущем задании,
#  с помощью кросс-валидации по 5 блокам с random_state = 42, не забудьте включить перемешивание выборки (shuffle=True).
gen = KFold(shuffle=True, n_splits=5, random_state=42)  # генератор разбиений


def metric_func(gen, x, y):
    scores = list()
    p_space = numpy.linspace(1, 10, 200)
    for i in p_space:
        gen = KFold(shuffle=True, n_splits=5, random_state=42)
        reg = KNeighborsRegressor(n_neighbors=5, weights='distance', p=i)
        tmp = cross_val_score(reg, x, y, cv=gen, scoring='neg_mean_squared_error')
        tmp = numpy.array(tmp).mean()
        scores.append(tmp)
    return pandas.DataFrame(scores, p_space).sort_values(by=[0], ascending=False)


metric_q = metric_func(gen, x, y)
print(metric_q)