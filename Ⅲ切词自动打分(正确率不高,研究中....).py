'''
切词，先学习一部分后进行标签打分
    打分标准:1-5分,仅根据社会阶层划分,有失偏额,仅供娱乐   分类数据,非连续
    1分:伤残,早逝,意外
    2分:温饱,低收入,十万
    3分:普通人,小康,百万
    4分:中产,千万,有名气
    5分:亿及以上
'''
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer # 词频统计包, 把评论内容 转成 词频矩阵
import jieba
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB               # 朴素贝叶斯对象
from sklearn.metrics import accuracy_score
import xgboost as xgb


#1.先尝试建立打分模型
def model_build():
    #读取数据集合
    csv = pd.read_csv('data/八字自动录入数据.csv',sep=',',encoding='gbk')
    print(csv)
    #读取当前csv文件中的所有评论内容,并使用jieba进行切割
    comment_lst = [','.join(list(jieba.cut(item))) for item in csv['评语']]
    # print(comment_list)
    #加载停用词表(网上找的),加上自己补的一些废话,删掉这些没用的词
    with open('data/stopwords.txt',mode='r',encoding='utf-8')as f:
        stopwords_lst = []
        for i in f:
            stopwords_lst.append(i.strip())
        stopwords_lst = list(set(stopwords_lst))    #去重
        # print(stopwords_lst)
    #建立切词对象,并根据切词统计词频
    transfer = CountVectorizer(stop_words=stopwords_lst)
    # x = transfer.fit_transform(comment_lst).toarray()          #查看一下二维矩阵长啥样,实际直接用稀疏矩阵就行
    # print(x)
    x = transfer.fit_transform(comment_lst)
    # print(list(transfer.get_feature_names_out()))          #看看有多少个词留下

    csv['得分'] = csv['得分'].map({'1':0,'2':1,'3':2,'4':3,'5':4}).astype('category')
    #取其中已评分的50条划分训练集和测试集
    x = x[:51]
    y = csv.iloc[:51,10]
    print(x)
    print(y.dtype)
    #划分测试集,训练集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=9,stratify=y)

    #建立朴素贝叶斯分类模型
    estimator = MultinomialNB()
    estimator.fit(x_train,y_train)
    y_pre = estimator.predict(x_test)
    print(f'正确率为{estimator.score(x_test,y_test)}')
    print(f'正确率为{accuracy_score(y_test,y_pre)}')


    #建立xgboost看看效果
    estimator1 = xgb.XGBClassifier(
        enable_categorical=True,
        max_depth= 3,
        n_estimators=100,
        random_state=6,
        objective='multi:softmax'
    )
    estimator1.fit(x_train,y_train)
    y_pre1 = estimator1.predict(x_test)
    print(f'正确率为{estimator1.score(x_test,y_test)}')
    print(f'正确率为{accuracy_score(y_test,y_pre1)}')

    #建立随机森林看看效果


if __name__ == '__main__':
    model_build()