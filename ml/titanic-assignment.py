# Предобработка данных в Pandas

import pandas
import numpy as np

data = pandas.read_csv('data/titanic.csv', index_col='PassengerId')

print(data['Sex'].value_counts())
print(data['Survived'].value_counts())
print(342 / (342 + 549)*100)
print(data['Pclass'].value_counts())
print(216 / (216 + 491 + 184)*100)
print(data['Age'].mean(), data['Age'].median())
print(data.corr(method='pearson'))

df = pandas.DataFrame(data=data, columns=['Name', 'Sex'])
df_fem = df.sort_values(by=['Sex'])[:314]
df_fem1 = pandas.DataFrame(data=df_fem, columns=['Name', 'Sex'])
name_list = df_fem1['Name'].get_values()
first_name_fem_list = []
first_name_fem_list2 = []
for i in range(0, name_list.size):
    str = name_list[i][:name_list[i].find(',')]
    first_name_fem_list.append(str)
for i in range(0, name_list.size):
    indx = name_list[i].find('(')
    if indx != -1:
        temp_str = name_list[i][name_list[i].find('(') + 1:]
        str = temp_str[:temp_str.find(' ')]
        first_name_fem_list2.append(str)
print(first_name_fem_list2)
first_name_fem_list = first_name_fem_list + first_name_fem_list2
for i in range(0, np.size(first_name_fem_list)):
    idx = first_name_fem_list[i].find('"')
    while idx != -1:
        first_name_fem_list[i] = first_name_fem_list[i].replace('"', ' ')
        idx = first_name_fem_list[i].find('"')
data = pandas.DataFrame(data=first_name_fem_list, columns=['Names'])
print(data['Names'].value_counts())
