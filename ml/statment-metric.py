import pandas
import numpy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve

# Загрузите файл classification.csv.
# В нем записаны истинные классы объектов выборки (колонка true) и ответы некоторого классификатора (колонка pred).

data = pandas.read_csv(r'C:\Users\olllk\Downloads\classification.csv')
data2 = pandas.read_csv(r'C:\Users\olllk\Downloads\scores.csv')
print(data)
true_col = numpy.array(data['true'])
pred_col = numpy.array(data['pred'])


# Для этого подсчитайте величины TP, FP, FN и TN согласно их определениям. Например, FP — это количество объектов,
# имеющих класс 0, но отнесенных алгоритмом к классу 1. Ответ в данном вопросе — четыре числа через пробел.
def table_fill(true_col, pred_col):
    tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
    for i in range(true_col.size):
        if (true_col[i] + pred_col[i]) == 2:
            tp += 1
        elif (true_col[i] + pred_col[i]) == 0:
            tn += 1
        elif true_col[i] > pred_col[i]:
            fn += 1
        else:
            fp += 1

    return [tp, fp, fn, tn]


arr = table_fill(true_col, pred_col)
print(arr)  # 43.0, 34.0, 59.0, 64.0
# Посчитайте основные метрики качества классификатора:

acc = accuracy_score(true_col, pred_col)
pr_sc = precision_score(true_col, pred_col)
rec_sc = recall_score(true_col, pred_col)
f1_sc = f1_score(true_col, pred_col)
print(acc, pr_sc, rec_sc, f1_sc)  # 0.535 0.5584415584415584 0.4215686274509804 0.48044692737430167

#  Посчитайте площадь под ROC-кривой для каждого классификатора. Какой классификатор имеет наибольшее
# значение метрики AUC-ROC (укажите название столбца)? Воспользуйтесь функцией sklearn.metrics.roc_auc_score.
print(data2)
auc_roc_1 = roc_auc_score(data2['true'], data2['score_logreg'])
auc_roc_2 = roc_auc_score(data2['true'], data2['score_svm'])
auc_roc_3 = roc_auc_score(data2['true'], data2['score_knn'])
auc_roc_4 = roc_auc_score(data2['true'], data2['score_tree'])
print(auc_roc_1, auc_roc_2, auc_roc_3, auc_roc_4)

# Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?
prec = numpy.array(precision_recall_curve(data2['true'], data2['score_logreg'])[0])[:197]
rec = numpy.array(precision_recall_curve(data2['true'], data2['score_logreg'])[1])[:197]
th = numpy.array(precision_recall_curve(data2['true'], data2['score_logreg'])[2])
tab = pandas.DataFrame({'precision': prec, 'rec': rec})
tab['new'] = th
tab = tab.sort_values(by=['precision'])
new_df = tab.apply(lambda wer: if tab['precision'] < 0.7 == 0)
# for a in range(tab[1].size):
