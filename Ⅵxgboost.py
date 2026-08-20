'''

'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV   #分开训练集和测试集的, 做交叉验证和网格搜索的
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # 数据标准化的
import xgboost as xgb                                   # 极限梯度提升树对象
from sklearn.metrics import accuracy_score                           # 模型评估的, 计算模型预测的准确率
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error,r2_score,classification_report   # 模型评估
import joblib                                                        #保存模型的
from sklearn.linear_model import Lasso, Ridge                    #L1正则化与L2正则化
from sklearn.utils import class_weight                           #权重平衡


def xgboosting():
    # 1.加载数据
    csv = pd.read_csv('data/八字自动录入数据(AI评分).csv', sep=',', usecols=['天干1', '地支1', '天干2', '地支2', '天干3', '地支3', '天干4', '地支4', '性别', 'AI评分'])
    # 2.数据预处理
    # data = csv.query('得分 not in ["争议",0,1,2,3,12,11,10,9]').loc[:, '天干1':'性别']
    data = csv.loc[:, '天干1':'性别']
    # target = csv.query('得分 not in ["争议",0,1,2,3,12,11,10,9]').loc[:, '得分'] - 4    #减去4,从0-4评分,不然会报错
    target = csv.loc[:, 'AI评分']    #减去4,从0-4评分,不然会报错
    # 3.特征工程
    # 3.1分类特征热编码(xgboost不用进行热编码)
    # data = pd.get_dummies(data,columns=['天干1', '地支1', '天干2', '地支2', '天干3', '地支3', '天干4', '地支4', '性别'],drop_first=True)  # drop_first删掉一个冗余的列
    # 3.2切分测试集训练集
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=9, stratify=target)
    # 4.模型训练
    sw = class_weight.compute_sample_weight('balanced', y_train)   #平衡权重
    estimator = xgb.XGBClassifier(
        enable_categorical=True,
        max_depth=3,                # 树的最大深度
        n_estimators=120,           # 树的数量
        learning_rate=0.01,          # 学习率
        objective='multi:softmax'   # 多分类问题, 使用多分类模型.
    )
    estimator.fit(x_train, y_train,sample_weight=sw)
    #5.模型评估
    y_pre = estimator.predict(x_test)
    print(f"分类评估报告\n{classification_report(y_test,y_pre)}")

if __name__ == '__main__':
    xgboosting()
