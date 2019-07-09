import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def sex_to_bin(x):
    if x == 'male':
        return 0
    elif x == 'female':
        return 1
    else:
        return x


data = pandas.read_csv('data/titanic.csv', index_col='PassengerId')
data = data.loc[data['Age'] is not np.nan and data['Age'] == data['Age']]  # deleting nan age data
features = pandas.DataFrame(data=data, columns=['Pclass', 'Fare', 'Age', 'Sex'])
features = features.applymap(sex_to_bin)
target = data['Survived'].get_values()
# print(features)

clf = DecisionTreeClassifier(random_state=241)
clf = clf.fit(features, target)

# noinspection PyUnresolvedReferences
importances = clf.feature_importances_
print(importances) # sex, fare