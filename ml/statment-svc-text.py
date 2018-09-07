from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas

# Загрузите объекты из новостного датасета 20 newsgroups, относящиеся к категориям "космос" и "атеизм"
# (инструкция приведена выше). Обратите внимание, что загрузка данных может занять несколько минут
newgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
X = newgroups.data
y = newgroups.target
# Вычислите TF-IDF-признаки для всех текстов. Обратите внимание, что в этом задании мы предлагаем вам вычислить
# TF-IDF по всем данным. При таком подходе получается, что признаки на обучающем множестве используют информацию
# из тестовой выборки — но такая ситуация вполне законна, поскольку мы не используем значения целевой переменной
#  из теста. На практике нередко встречаются ситуации, когда признаки объектов тестовой выборки известны
# на момент обучения, и поэтому можно ими пользоваться при обучении алгоритма.
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(X)
# Подберите минимальный лучший параметр C из множества [10^-5, 10^-4, ... 10^4, 10^5] для SVM с линейным ядром
# (kernel='linear') при помощи кросс-валидации по 5 блокам. Укажите параметр random_state=241 и для SVM, и для
# KFold. В качестве меры качества используйте долю верных ответов (accuracy).
arr = np.array(range(-5, 6))
grid = {'C': np.float_power(10, arr)}
cv = KFold(n_splits=5, shuffle=True, random_state=241)  # генератор разбиений
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(vectorizer.transform(X), y)
