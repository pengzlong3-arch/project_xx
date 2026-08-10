'''

'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV   #分开训练集和测试集的, 做交叉验证和网格搜索的
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier                   # KNN算法 分类对象
from sklearn.neighbors import KNeighborsRegressor                    # KNN算法的 回归模型
from sklearn.metrics import accuracy_score                           # 模型评估的, 计算模型预测的准确率
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error,r2_score,classification_report   # 模型评估
import joblib                                                        #保存模型的
from sklearn.linear_model import Lasso, Ridge                    #L1正则化与L2正则化

def knn():
    #1.加载数据
    csv = pd.read_csv('data/八字数据.csv', sep=',', usecols=['天干1', '地支1', '天干2', '地支2', '天干3', '地支3', '天干4', '地支4', '性别', '得分'])
    #2.数据预处理
    data = csv.query('得分 not in ["争议",0,1,2,3,12,11,10,9]').loc[:, '天干1':'性别']
    target = csv.query('得分 not in ["争议",0,1,2,3,12,11,10,9]').loc[:, '得分']
    #3.特征工程
    #3.1分类特征热编码(仅做探索性测试,应该在划分训练集之后再做)
    data = pd.get_dummies(data,columns=['天干1', '地支1', '天干2', '地支2', '天干3', '地支3', '天干4', '地支4', '性别'],drop_first=True)  # drop_first删掉一个冗余的列
    #3.2切分测试集训练集
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=11,stratify=target)
    #4.模型训练
    #4.1创建KNN分类对象
    estimator = KNeighborsClassifier(n_neighbors=3)
    #4.1.1模型训练
    estimator.fit(x_train,y_train)
    y_pre = estimator.predict(x_test)

    #4.2创建进行网格搜索&交叉验证的对象
    estimator2 = KNeighborsClassifier()
    #4.2.1网格搜索&交叉验证
    estimator_lst = GridSearchCV(estimator2, {'n_neighbors': [i for i in range(3, 11)]}, cv=5, error_score='raise') #训练失败的话报错
    estimator_lst.fit(x_train,y_train)
    #5.模型评估
    #5.1单一模型评估
    print(f'单独模型的正确率是{accuracy_score(y_test,y_pre)}')
    #5.2网格搜索&交叉验证结果
    print(f'最佳验证的评分{estimator_lst.best_score_}')
    print(f'最佳验证的超参数{estimator_lst.best_params_}')
    print(f'最佳验证的结果{estimator_lst.cv_results_}')

if __name__ == '__main__':
    knn()