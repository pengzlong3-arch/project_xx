


def auto_in():
    lst = []
    while 1:
        res = input('请输入性别+八字:')
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

        except Exception as e:
            print(f'数据录入失败:{e}')
        print(lst)

auto_in()
