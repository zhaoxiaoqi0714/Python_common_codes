#### Randorm forest 参数训练

from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor  
import numpy as np  
import pandas as pd

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

### 2. 决策树建模
import sklearn.tree as tree

# 直接使用交叉网格搜索来优化决策树模型，边训练边优化

# 网格搜索的参数：正常决策树建模中的参数 - 评估指标，树的深度，
 ## 最小拆分的叶子样本数与树的深度
param_grid = {'criterion': ['entropy', 'gini'],
             'max_depth': [2, 3, 4, 5, 6, 7, 8],
             'min_samples_split': [4, 8, 12, 16, 20, 24, 28]} 
                # 通常来说，十几层的树已经是比较深了

clf = tree.DecisionTreeClassifier()  # 定义一棵树
clfcv = GridSearchCV(estimator=clf, param_grid=param_grid, 
                            scoring='roc_auc', cv=4) 
        # 传入模型，网格搜索的参数，评估指标，cv交叉验证的次数
      
clfcv.fit(X=X_train, y=y_train)

# 使用模型来对测试集进行预测
test_est = clfcv.predict(X_test)

# 模型评估
import sklearn.metrics as metrics

print("决策树准确度:")
print(metrics.classification_report(y_test,test_est)) 
        # 该矩阵表格其实作用不大
print("决策树 AUC:")
fpr_test, tpr_test, th_test = metrics.roc_curve(y_test, test_est)
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

### 3. 随机森林建模
param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[5, 6, 7, 8],    # 深度：这里是森林中每棵决策树的深度
    'n_estimators':[11,13,15],  # 决策树个数-随机森林特有参数
    'max_features':[0.3,0.4,0.5],
     # 每棵决策树使用的变量占比-随机森林特有参数（结合原理）
    'min_samples_split':[4,8,12,16]  # 叶子的最小拆分样本量
}

import sklearn.ensemble as ensemble # ensemble learning: 集成学习

rfc = ensemble.RandomForestClassifier()
rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid,
                      scoring='roc_auc', cv=4)
rfc_cv.fit(X_train, y_train)

# 使用随机森林对测试集进行预测
test_est = rfc_cv.predict(X_test)
print('随机森林精确度...')
print(metrics.classification_report(test_est, y_test))
print('随机森林 AUC...')
fpr_test, tpr_test, th_test = metrics.roc_curve(test_est, y_test)
     # 构造 roc 曲线
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

#### 可以根据结果对param_grid调整参数，使设定参数范围内出现拐点，确定最优参数，提高模型精度

### 4. 特征值筛选
from sklearn.model_selection import cross_val_score, ShuffleSplit  

rf = RandomForestRegressor()  
scores = []  
for i in range(iris["data"].shape[1]):  
     score = cross_val_score(rf, iris["data"][:, i:i+1],iris["target"], scoring="r2",  
                              cv=ShuffleSplit(len(iris["data"]), 3, .3))  
     scores.append((round(np.mean(score), 3), iris["feature_names"][i]))  
     
print(sorted(scores, reverse=True))


### 5. multi-fold cross-validation
# optimized parameters
param_grid={'criterion': ['entropy', 'gini'],
                         'max_depth': 5,
                         'max_features': 0.4,
                         'min_samples_split': 2,
                         'n_estimators': 100}

rfc = ensemble.RandomForestClassifier()
rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid,
                      scoring='roc_auc', cv=5) # cross_validation
rfc_cv.fit(X_train, y_train)

# 使用随机森林对测试集进行预测
test_est = rfc_cv.predict(X_test)
print('随机森林精确度...')
print(metrics.classification_report(test_est, y_test))
print('随机森林 AUC...')
fpr_test, tpr_test, th_test = metrics.roc_curve(test_est, y_test)
     # 构造 roc 曲线
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

###################################################
#### oob test
## 计算特征值重要度变化， 通过利用oob对每个特征分别permutation或加噪音，迭代进行，评估改变每个特征后的∑(ei2-e1)/T，然后做一个排序，值越高的特征越重要。
import copy 
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

### 1. load data
from sklearn.datasets import load_iris  
from sklearn.model_selection import cross_val_score, ShuffleSplit, train_test_split, GridSearchCV
iris=load_iris()

data = pd.concat([pd.DataFrame(iris["data"]), pd.DataFrame(iris['target'])], axis = 1)
data = data[data.iloc[:,4].isin([0,1])]

x = np.array(data.iloc[:,0:4])
y = np.array(data.iloc[:,4])
feature_names = iris["feature_names"]

X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                    test_size=0.4, random_state=12345)

### 2. constructing model
rf = RandomForestClassifier(n_estimators=200,oob_score=True)
rf.fit(X_train,y_train)
features_imp = rf.feature_importances_

def select_combine(X_train,y_train,features_name,features_imp,select_num):
    oob_result = []
    fea_result = []
    features_imp = list(features_imp)
    iter_count = X_train.shape[1] - select_num  #迭代次数
    if iter_count < 0:
        print("select_nume must less or equal X_train columns")
    else:
        features_test = copy.deepcopy(features_imp)   #生成一个排序特征，进行筛选
        features_test.sort()
        features_test.reverse() 
        
        while iter_count >= 0:
            iter_count -= 1
            train_index = [features_imp.index(j) for j in features_test[:select_num]]
            train_feature_name = [features_name[k] for k in train_index][0]
            train_data = X_train[:,train_index]
            rf.fit(train_data,y_train)
            acc = rf.oob_score_
            print(acc)
            oob_result.append(acc)
            fea_result.append(train_index)
            if select_num < X_train.shape[1]:
                select_num += 1
            else:
                break
    return max(oob_result),oob_result,fea_result[oob_result.index(max(oob_result))]
 
select_num = 3
max_result, oob_result, fea_result = select_combine(X_train,y_train,feature_names,features_imp,select_num)












