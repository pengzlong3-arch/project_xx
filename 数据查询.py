import pandas as pd

# df = pd.read_csv('八字自动录入数据(已评分).csv',sep=',', usecols= ['天干1','天干2',
#                                                          '天干3','天干4','地支1','地支2',
#                                                          '地支3','地支4','性别','合并','得分','评语'])

df = pd.read_csv('data/八字数据.csv', sep=',', usecols= ['天干1', '天干2',
                                                         '天干3','天干4','地支1','地支2',
                                                         '地支3','地支4','性别','合并','得分','评语'])
def search_label():       #查看标签分布情况的
    df1 = df['得分'].value_counts().sort_index()
    print(df1)
    print(df.shape[0])

search_label()