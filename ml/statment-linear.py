from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy
import pandas

# Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и perceptron-test.csv.
#  Целевая переменная записана в первом столбце, признаки — во втором и третьем.
data_train = pandas.read_csv(r'C:\Users\olllk\Downloads\perceptron-train.csv', header=None)
data_test = pandas.read_csv(r'C:\Users\olllk\Downloads\perceptron-test.csv', header=None)
X_train = data_train.loc[:, 1:]
y_train = data_train[0]
X_test = data_test.loc[:, 1:]
y_test = data_test[0]

# Обучите персептрон со стандартными параметрами и random_state=241.
clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

# Подсчитайте качество (долю правильно
# классифицированных объектов, accuracy) полученного классификатора на тестовой выборке.
ac_sc = accuracy_score(y_test, pred)
print(ac_sc)
# Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler.
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Обучите персептрон на новой выборке. Найдите долю правильных ответов на тестовой выборке.
clf.fit(X_train_scaled, y_train)
pred = clf.predict(X_test_scaled)

# Найдите разность между качеством на тестовой выборке после нормализации и качеством до нее.
# Это число и будет ответом на задание.
ac_sc_1 = accuracy_score(y_test, pred)
difference = ac_sc_1 - ac_sc
print(difference)
