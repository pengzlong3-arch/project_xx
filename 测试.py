import pandas as pd

csv = pd.read_csv('八字自动录入数据(已评分).csv',sep=',',usecols=['天干1','地支1','天干2','地支2','天干3','地支3','天干4','地支4','性别','得分'])
print(csv)