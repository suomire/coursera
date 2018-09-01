import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import scipy.stats as ss

data = pd.read_csv(r'C:\Users\olllk\Downloads\data.csv', index_col='PassengerId')
data_ = data['Age']
data_ = data_.dropna()
data_ = data_[np.isfinite(data_)]
data_array = np.array(data_)

m = int(data_array.size ** 0.5)
data_array.sort()
min_value = min(data_array)
max_value = max(data_array)

distribution_fun = np.zeros(m)
h = (max_value - min_value) / m
steps = []
for t in range(1, m + 1):
    steps.append(min_value + t * h)

index = 0
for value in data_array:
    if value > steps[index]:
        p = int(abs(steps[index] - value) // h) + 1
        for i in range(1, p):
            distribution_fun[index + i] = distribution_fun[index]
        index += p
        distribution_fun[index] = distribution_fun[index - 1]
    distribution_fun[index] += 1

plot.title("Функция распределения")
# plot.xlim([0.6, 13])
plot.bar(steps, distribution_fun / data_array.size)
plot.show()
plot.title("Гистограмма")
plot.hist(data_array, steps, density=1)

# ------------------------------------------------erlang----------------------------------------

k = 1.5

print(ss.gamma.fit(data_array, k))
shape, loc_e, lambd_1 = ss.gamma.fit(data_array, k)
x = np.linspace(min(data_array), max(data_array), data_array.size)
distr_data = ss.erlang.pdf(x, shape, loc=loc_e, scale=lambd_1)
plot.plot(x, distr_data, 'r-', alpha=1)
# koef_1 = ss.pearsonr(data_array, x)
# print(koef_1)


# a = input()
# plot.show()
# -----------------------------------------------student-----------------------------------------------
df = 120
print(ss.t.fit(data_array, df))
t_mp, loc1, lambd_2 = ss.t.fit(data_array, df)
x = np.linspace(min(data_array), max(data_array), data_array.size)
distr_data = ss.t.pdf(x, df, loc=loc1, scale=lambd_2)
plot.plot(x, distr_data, 'y-', alpha=1)

# -----------------------------------------norm---------------------------------------------

# ---------------------------------koef variacii--------------------------------------------------------------------
n = m
mean = np.sum(data_array) / n
print("mean = {0}".format(mean))
S = np.sqrt(np.sum((data_array - np.mean(data_array)) ** 2) / n - 1)
print("S = {0}".format(S))
V_x = S / mean
print("V_x = {0}".format(V_x))
# -------------------------------------------pearson-----------------------------------------------------------------
counts, bins = np.histogram(data_array, bins=steps)
p_observed = np.array(counts, dtype=float)
p_observed_1 = np.array(counts, dtype=float)
mm = np.sum(p_observed) / p_observed.size
sttd = np.std(p_observed)
p_norm = ((p_observed - m) / sttd)
print(mm, sttd, p_norm)
p_observed=p_norm

tmp = ss.erlang.cdf(bins, shape, loc=loc_e, scale=lambd_1)
ttmp = ss.t.cdf(bins, t_mp, loc=loc1, scale=lambd_2)
print(tmp, ttmp)

# Now get probabilities for intervals
p_expected = []
for idx, p in enumerate(tmp):
    if idx == len(tmp) - 1: break
    print(idx, p)
    p_expected.append(tmp[idx + 1] - tmp[idx])
print(p_expected)
p_expected = data_array.size * np.array(p_expected)
print(p_observed, p_expected)
plot.show()
p_expected_1 = []
for idx, p in enumerate(ttmp):
    if idx == len(ttmp) - 1: break
    p_expected_1.append(ttmp[idx + 1] - ttmp[idx])

p_expected_1 = data_array.size * np.array(p_expected_1)

# Calculate using scipy built in chi-square test
chi_test_value, probability_fit = ss.chisquare(p_observed, np.array(p_expected, dtype=float))
chi_test_value_1, probability_fit_1 = ss.chisquare(p_observed_1, np.array(p_expected_1, dtype=float))
print('Вероятность того, что распределение Эрланга подходит к полученным данным:', probability_fit)
print('Вероятность того, что распределение Стьюдента подходит к полученным данным:', probability_fit_1)

# Calculate by ourselved
chi_star = np.sum((p_observed - p_expected) ** 2 / p_expected)
chi_star_1 = np.sum((p_observed_1 - p_expected_1) ** 2 / p_expected_1)
print("chi_star = {}".format(chi_star))
print("chi_star_1 = {}".format(chi_star_1))
conf_interval = 0.95
df = len(bins) - 3
chi = ss.chi2.ppf(conf_interval, df)  # chi-square quntile for alpha = conf-interval, degrees of freedom = df
print("chi = {}".format(chi))

a = int(input())
print(min_value)
fdata = data
fdata = fdata.loc[fdata['Sex'] != 'male']
print(fdata)
fdata['Surname'] = fdata['Name'].str.split(',').str.get(0)
fdata['First Name'] = fdata['Name'].str.split(',').str.get(1)

newfdata = fdata[fdata.columns[11:13]]
print(newfdata)
c = []
for i in newfdata['First Name']:
    if '(' in i:
        if ')' in i.split('(')[1].split(' ')[0]:
            c.append(i.split('(')[1].split(' ')[0].split(')')[0])
        else:
            i.split('(')[1].split(' ')[0]
    else:
        c.append(i.split('. ')[1].split(' ')[0])
print(c)
print(pd.DataFrame.from_dict(c)[0].value_counts())
