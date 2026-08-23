'''

'''
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset      # 数据集对象.   数据 -> Tensor -> 数据集 -> 数据加载器
from torch.utils.data import DataLoader      # 数据集对象.   数据 -> Tensor -> 数据集 -> 数据加载器
from sklearn.model_selection import train_test_split, GridSearchCV   #分开训练集和测试集的, 做交叉验证和网格搜索的


def create_data():
    df = pd.read_csv('data/八字自动录入数据(AI评分).csv',sep=',',encoding='utf-8')
    print(df)

    x,y = df.iloc[:,0:9],df.iloc[:,10]
    # print(x.dtypes,y.dtypes)   #发现是str,要改
    # x.info()            #描述也显示是info


    #划分训练集测试集
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=5,stratify=y)

    # x_train.info()    #全是str得改
    x_train.astype('float32')
    x_test.astype('float32')
    y_train.astype('float32')
    y_test.astype('float32')

    #转成torch分批加载对象
    train_dataset = TensorDataset(torch.tensor(x_train.values,dtype=torch.float32),torch.tensor(y_train.values,dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(x_test.values,dtype=torch.float32),torch.tensor(y_test.values,dtype=torch.long))

    return train_dataset, test_dataset



class Predict_model(nn.Module):
    def __init__(self):
        super().__init__()
        #搭建神经网络全连接层
        #第一层
        self.linear1 = nn.Linear(9,128)
        #初始化权重
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        #第二层
        self.linear2 = nn.Linear(128,128)
        #初始化权重
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

        #第三层
        self.linear3 = nn.Linear(128,256)
        #初始化权重
        nn.init.kaiming_normal_(self.linear3.weight)
        nn.init.zeros_(self.linear3.bias)

        #第四层
        self.linear4 = nn.Linear(256,128)
        #初始化权重
        nn.init.kaiming_normal_(self.linear4.weight)
        nn.init.zeros_(self.linear4.bias)

        self.output = nn.Linear(128,5)

        #随机失活
        self.dropout = nn.Dropout(0.2)
    #定义前向传播方法
    def forward(self, x):         #forward是自动调用的
        #隐藏层1
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)

        #隐藏层2
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)

        #隐藏层3
        x = torch.relu(self.linear3(x))
        x = self.dropout(x)

        #隐藏层4
        x = torch.relu(self.linear4(x))
        x = self.dropout(x)

        #输出层
        x = self.output(x)
        return x

def train():
    #固定随机种子
    torch.manual_seed(5)

    #获取训练数据集
    train_dataset,test_dataset = create_data()

    #初始化数据加载器
    dataloader =  DataLoader(train_dataset,batch_size=16,shuffle=True)

    #初始化模型
    model = Predict_model()

    #初始化损失函数
    criterion = nn.CrossEntropyLoss()

    #梯度优化方法
    optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999))

    #遍历每轮的数据
    epoch = 100
    for epo in range(epoch):
        #训练时间
        start_time = time.time()
        #计算损失
        total_loss = 0.0
        total_num = 0.0
        #遍历每个batch进行处理
        for x,y in dataloader:
            model.train()
            output = model(x)

            #计算损失
            loss = criterion(output,y)
            #梯度清零
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #参数更新
            optimizer.step()
            #损失计算
            total_num += len(y)
            total_loss += loss * len(y)
        #打印损失计算结果
        print(f'训练轮数:{epo+1},损失:{(total_loss/total_num):.2f},时间:{(time.time()-start_time):.2f}')
    torch.save(model.state_dict(),'model/predict.pth')

#模型评估
def evaluate():
    train_dataset,test_dataset = create_data()

    #使用之前搭建的神经网络结构
    model = Predict_model()

    #加载模型存下来的参数
    model.load_state_dict(torch.load('model/predict.pth'))

    #构建数据加载器
    dataloader = DataLoader(test_dataset,shuffle=True,batch_size=19)

    #评估数据集
    correct = 0
    for x,y in dataloader:
        model.eval()

        output = model(x)

        y_pre = torch.argmax(output,dim=1)   #按1维找,相当于找行最大的
        print(y_pre)

        correct += (y_pre == y).sum()

        print(f'正确率{correct:.2f}%')




if __name__ == '__main__':
    #模型训练
    train()
    #模型测试
    evaluate()