'''

'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # 数据标准化的
from sklearn.linear_model import LogisticRegression                   #逻辑回归
from sklearn.metrics import classification_report                    # 模型评估
import joblib                                                        #保存模型的

def logisticregression():
    # 1.加载数据
    csv = pd.read_csv('data/八字数据.csv', sep=',', usecols=['天干1', '地支1', '天干2', '地支2', '天干3', '地支3', '天干4', '地支4', '性别', '得分'])
    # 2.数据预处理
    data = csv.query('得分 not in ["争议",0,1,2,3,12,11,10,9]').loc[:, '天干1':'性别']
    target = csv.query('得分 not in ["争议",0,1,2,3,12,11,10,9]').loc[:, '得分']
    # 3.特征工程
    # 3.1分类特征热编码
    data = pd.get_dummies(data, columns=['天干1', '地支1', '天干2', '地支2', '天干3', '地支3', '天干4', '地支4', '性别'],drop_first=True)  # drop_first删掉一个冗余的列
    # 3.2切分测试集训练集
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=11, stratify=target)
    # 4.模型训练
    estimator = LogisticRegression(solver='lbfgs',class_weight='balanced')
    estimator.fit(x_train,y_train)
    #5.模型评估
    y_pre = estimator.predict(x_test)
    print(f'分类评估报告\n{classification_report(y_test,y_pre)}')

if __name__ == '__main__':
    logisticregression()