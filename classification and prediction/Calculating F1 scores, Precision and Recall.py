### calculating F1 scores, Precision and Recall

from sklearn.metrics import f1_score, precision_score, recall_score

y_true=[1,2,3]
y_pred=[1,1,3]

f1 = f1_score(y_true, y_pred, average='macro' )
p = precision_score(y_true, y_pred, average='macro')
r = recall_score(y_true, y_pred, average='macro')

print(f1, p, r)
# output: 0.555555555556 0.5 0.666666666667








