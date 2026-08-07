'''
数字代表类别
录入数据: 甲 乙 丙 丁 戊 己 庚 辛 壬 癸
         1 2  3  4  5  6 7  8 9 10
录入数据: 子 丑 寅 卯 辰 巳 午 未 申 酉 戌 亥
         1 2  3  4 5  6 7  8  9 10 11 12
性别：男1，女2
'''
import pandas as pd
import os


def exist():
#创建数据文件
    if os.path.exists('data/八字数据.csv'):
        csv = pd.read_csv('data/八字数据.csv', sep=',', usecols= ['天干1', '地支1',
                                                    '天干2','地支2','天干3','地支3',
                                                    '天干4','地支4','性别','合并','得分','评语'])
        # print(csv)
        print('-'*23)
        print(csv.shape)
        print('-'*23)
    else:
        csv = pd.DataFrame([], columns= ['天干1','地支1',
                                                    '天干2','地支2','天干3','地支3',
                                                    '天干4','地支4','性别','合并','得分','评语'])
        csv.to_csv('八字数据.csv', sep=',', index=False)



def save(x):
    lst = pd.DataFrame([x], columns= ['天干1','地支1',
                                                    '天干2','地支2','天干3','地支3',
                                                    '天干4','地支4','性别','合并','得分','评语'])
    lst.to_csv('八字数据.csv', sep=',',header=False, index= False, mode='a')

def type_in():
    while 1 :
        lst = []
        res = str(input('请输入八字天干地支(quit退出):\n').strip().replace(' ',''))  #去除空白
        if res.upper().strip() == 'QUIT':
            return
        try:   #甲辰甲辰甲辰甲辰
            item = ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸",'子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥']
            res_set = set(res)   #去重复
            if all(i in item for i in res_set):
                for n in range(0,len(res),2):
                    i = res[n]
                    j = res[n+1]
                    match i:
                        case '甲':lst.append(1)
                        case '乙':lst.append(2)
                        case '丙':lst.append(3)
                        case '丁':lst.append(4)
                        case '戊':lst.append(5)
                        case '己':lst.append(6)
                        case '庚':lst.append(7)
                        case '辛':lst.append(8)
                        case '壬':lst.append(9)
                        case '癸':lst.append(10)
                    match j:
                        case '子':lst.append(1)
                        case '丑':lst.append(2)
                        case '寅':lst.append(3)
                        case '卯':lst.append(4)
                        case '辰':lst.append(5)
                        case '巳':lst.append(6)
                        case '午':lst.append(7)
                        case '未':lst.append(8)
                        case '申':lst.append(9)
                        case '酉':lst.append(10)
                        case '戌':lst.append(11)
                        case '亥':lst.append(12)

                lst.append(res)    #合并
                while 1:
                    try:
                        res = input('请输入性别(1男_2女):\n')
                        if res.upper().strip() == 'QUIT':
                            return
                        gender = int(res)
                        if gender in [1,2]:
                            lst.append(gender)
                            break
                        else:
                            print('输入有误,请重新输入')
                    except:
                        print('输入性别有误')
                while 1:
                    try:
                        res = input('请输入得分(0-2早夭,3-5比普通人差点,6普通人,7-9不是普通人,10-12佼佼者):\n')
                        if res.upper().strip() == 'QUIT':
                            return
                        score = int(res)
                        lst.append(score)
                        break
                    except:
                        print('输入分数错误')
                while 1:
                    try:
                        comment = input('请输入评语:\n')
                        if comment.upper().strip() == 'QUIT':
                            return
                        lst.append(comment)
                        break
                    except:
                        print('输入评语有误')
                # print(lst)
                save(lst)  # 全部装进后保存

            else:
                print('输入八字有误')
        except Exception as e:
            print(f'输入错误{e}\n')

        # print(lst)
        # print(hole_lst)

def auto_in(res, res_all):
    print('正在录入数据')
    lst = []
    try:
        item = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸", '子', '丑', '寅', '卯', '辰', '巳', '午',
                '未', '申', '酉', '戌', '亥']
        res_set = set(res[1:])  # 去重复
        res_name = res[1:]
        print(res_name)
        if all(i in item for i in res_set):
            for n in range(0, len(res_name), 2):
                i = res_name[n]
                j = res_name[n + 1]
                match i:
                    case '甲':
                        lst.append(1)
                    case '乙':
                        lst.append(2)
                    case '丙':
                        lst.append(3)
                    case '丁':
                        lst.append(4)
                    case '戊':
                        lst.append(5)
                    case '己':
                        lst.append(6)
                    case '庚':
                        lst.append(7)
                    case '辛':
                        lst.append(8)
                    case '壬':
                        lst.append(9)
                    case '癸':
                        lst.append(10)
                match j:
                    case '子':
                        lst.append(1)
                    case '丑':
                        lst.append(2)
                    case '寅':
                        lst.append(3)
                    case '卯':
                        lst.append(4)
                    case '辰':
                        lst.append(5)
                    case '巳':
                        lst.append(6)
                    case '午':
                        lst.append(7)
                    case '未':
                        lst.append(8)
                    case '申':
                        lst.append(9)
                    case '酉':
                        lst.append(10)
                    case '戌':
                        lst.append(11)
                    case '亥':
                        lst.append(12)
            lst.append(1) if res[0] == '乾' else lst.append(2)    #输入性别

        lst.append(res)                                          #输入合并
        lst.append('未评分')                                      #未评分
        lst.append(res_all)                                      #输入评语

    except Exception as e:
        print(f'数据录入失败:{e}')
    # print(lst)
    auto_save(lst)



def auto_save(x):
    lst_all = []
    lst_all.append(x)
    lst = pd.DataFrame(lst_all, columns= ['天干1','地支1',
                                                    '天干2','地支2','天干3','地支3',
                                                    '天干4','地支4','性别','合并','得分','评语'])
    if os.path.exists('data/八字自动录入数据.csv'):
        lst.to_csv('八字自动录入数据.csv', sep=',', index= False, mode='a',header=False)
    else:
        lst.to_csv('八字自动录入数据.csv', sep=',', index= False, mode='w',header=True)
    print('已成功录入')

if __name__ == '__main__':
    exist()
    type_in()
