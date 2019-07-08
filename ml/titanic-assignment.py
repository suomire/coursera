# Предобработка данных в Pandas

import pandas

data = pandas.read_csv('data/titanic.csv', index_col='PassengerId')
"""
print(data['Sex'].value_counts())
print(data['Survived'].value_counts())
print(342 / (342 + 549))
print(data['Pclass'].value_counts())
print(216 / (216 + 491 + 184))
print(data['Age'].mean(), data['Age'].median())
print(data.corr(method='pearson'))
"""
df = pandas.DataFrame(data=data, columns=['Name', 'Sex'])
df_fem = df.sort_values(by=['Sex'])[:314]
df_fem1 = pandas.DataFrame(data=df_fem, columns=['Name', 'Sex'])
name_list = df_fem1['Name'].get_values()
print(name_list[1])
