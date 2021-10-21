#### logistic, The decision tree, SVM, Random forest....(cross validation)

import numpy as np   
import pandas as pd
import warnings
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

### 1. load data
from sklearn.datasets import load_iris  
from sklearn.model_selection import cross_val_score, ShuffleSplit, train_test_split, GridSearchCV
iris=load_iris()

data = pd.concat([pd.DataFrame(iris["data"]), pd.DataFrame(iris['target'])], axis = 1)
data = data[data.iloc[:,4].isin([0,1])]

x = data.iloc[:,0:4]
y = data.iloc[:,4]

X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                    test_size=0.4, random_state=12345)

### 2. construct model
lr = LogisticRegression(random_state=777,tol=1e-6)  # 逻辑回归模型
tree = DecisionTreeClassifier(random_state=777) #决策树模型
svm = SVC(probability=True,random_state=777,tol=1e-6)  # SVM模型
forest=RandomForestClassifier(n_estimators=100,random_state=777) #　随机森林
Gbdt=GradientBoostingClassifier(random_state=777) #CBDT

### 3. 构建评分函数，采取5折交叉验证的方式评分

def muti_score(model):
    warnings.filterwarnings('ignore')
    accuracy = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5)
    precision = cross_val_score(model, X_train, y_train, scoring='precision', cv=5)
    recall = cross_val_score(model, X_train, y_train, scoring='recall', cv=5)
    f1_score = cross_val_score(model, X_train, y_train, scoring='f1', cv=5)
    auc = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=5)
    print("准确率:",accuracy.mean())
    print("精确率:",precision.mean())
    print("召回率:",recall.mean())
    print("F1_score:",f1_score.mean())
    print("AUC:",auc.mean())


### 4. training
model_name=["lr","tree","svm","forest","Gbdt"]
for name in model_name:
    model=eval(name)
    print(name)
    muti_score(model)






