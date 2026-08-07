'''

'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV   #分开训练集和测试集的, 做交叉验证和网格搜索的
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # 数据标准化的
# from sklearn.neighbors import KNeighborsClassifier                   # KNN算法 分类对象
from sklearn.linear_model import LinearRegression                 # KNN算法 分类对象
from sklearn.neighbors import KNeighborsRegressor                    # KNN算法的 回归模型
# from sklearn.metrics import accuracy_score                           # 模型评估的, 计算模型预测的准确率
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error,r2_score,classification_report   # 模型评估
from collections import Counter                                      #用来查看标签分布情况的
import joblib                                                        #保存模型的
from sklearn.linear_model import Lasso, Ridge                    #L1正则化与L2正则化
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

csv = pd.read_csv('data/八字数据.csv', sep=',', usecols=['天干1', '地支1', '天干2', '地支2', '天干3', '地支3', '天干4', '地支4', '性别', '得分'])
# csv = pd.read_csv('八字自动录入数据(已评分).csv',sep=',',usecols=['天干1','地支1','天干2','地支2','天干3','地支3','天干4','地支4','性别','得分'])
print(csv)
data = csv.query('得分 not in ["争议",0,1,2,3,12,11,10,9]').loc[:,'天干1':'性别']
target = csv.query('得分 not in ["争议",0,1,2,3,12,11,10,9]').loc[:,'得分'] -4
# print(data.shape)
# print(target.shape)
print(data)
print(target)
#热编码处理
data = pd.get_dummies(data,columns=['天干1','地支1','天干2','地支2','天干3','地支3','天干4','地支4','性别'],drop_first=True) #dropfirst删掉一个冗余的列
print(data)
data_all = np.hstack([data])
# print(data_all)

#分离测试集和训练集
x_train,x_test,y_train,y_test = train_test_split(data_all,target,test_size=0.2,random_state=11,stratify=target)
print(x_train)

# #创建标准化的对象
# std = StandardScaler()
# #fit_transform用来标准化训练集的特征数据
# x_train_std = std.fit_transform(x_train)
# #用transform来标准化测试集的特征数据
# x_test_std = std.transform(x_test)

# #下面部分先暂时不做
# #创建KNN分类对象
# estimator = KNeighborsClassifier()
# #交叉验证+网格分析
# estimator_lst = GridSearchCV(estimator,{'n_neighbors':[i for i in range(1,11)]},cv=7,error_score='raise')
# #进行模型拟合
# estimator_lst.fit(x_train_std,y_train)
# #获取拟合的最好评分&建议的最好超参数N值
# print(f'最优的评分{estimator_lst.best_score_}')
# print(f'最优的评分{estimator_lst.best_params_}')
# print(f'具体的交叉验证结果: {estimator_lst.cv_results_}')


#直接建立模型,并进行训练
#线性模型，树模型
# estimator_model = LinearRegression(fit_intercept=True)
# estimator_model = Ridge(fit_intercept=True,alpha=0.1)
# estimator_model = DecisionTreeRegressor(max_depth=18)
# estimator_model.fit(x_train,y_train)
#集成学习: 自适应提升、GBDT(梯度提升树)、xgboost
# estimator_model2 = AdaBoostRegressor(estimator=estimator_model,n_estimators=80,learning_rate=0.1)
# estimator_model2 = GradientBoostingRegressor(n_estimators=120,learning_rate=0.025,max_depth=3)
estimator_model2 = xgb.XGBClassifier(n_estimators=100,max_depth=5,learning_rate=0.1,objective='multi:softmax',random_state=22)
estimator_model2.fit(x_train,y_train)


# #测试
# y_pre = estimator_model.predict(x_test_std)
# y_pre = estimator_model.predict(x_test)
y_pre = estimator_model2.predict(x_test)
# MSE = mean_squared_error(y_test,y_pre)
# RMSE = root_mean_squared_error(y_test,y_pre)
# MAE = mean_absolute_error(y_test,y_pre)
# r2 = r2_score(y_test,y_pre)
# print(MSE)
# print(RMSE)
# print(MAE)
# print(r2)
print(f'分类评估报告{classification_report(y_test,y_pre)}')
print(y_pre)
print('-'*23)
print(y_test.to_numpy())
#
# #输入预测
# X = np.array([9,3,5,9,8,6,5,11,1])
# # #热编码处理
# # X = pd.get_dummies(X,drop_first=True)
# # X_true = np.hstack([X,X**2,X**3])
# # print(X_true)
# # # X_true_std = std.transform(X_true)
# # # Y_pre = estimator_model.predict(X_true_std)
# # Y_pre = estimator_model.predict(X_true)
# # # print(estimator_model.coef_)
# # # print(estimator_model.intercept_)
# # print(Y_pre)