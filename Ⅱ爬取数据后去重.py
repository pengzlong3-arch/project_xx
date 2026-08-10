from tkinter.constants import FIRST

import pandas as pd

df = pd.read_csv('data/八字自动录入数据.csv', sep=',')
df.drop_duplicates(inplace=True,ignore_index=True,subset='评语',keep='first')    #评语一致的才删掉,保留最先出现的
print(df)
df.to_csv('data/八字自动录入数据.csv', sep=',',header=True, index= False)