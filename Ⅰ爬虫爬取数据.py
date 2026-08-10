'''
爬取大龙盲派的公众号数据
'''
import re
import time
import requests
from lxml import html
from 手动数据录入 import auto_in

class Robot():
    def __init__(self):
        self.url = 'https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzYyNDkwNzI0Mg==&action=getalbum&album_id=4327161721590038539&subscene=159&subscene=&scenenote=https%3A%2F%2Fmp.weixin.qq.com%2Fs%2F4zYi45etCPP9-4RHwra6AQ&nolastread=1#wechat_redirect'
        self.headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36 Edg/150.0.0.0',
                        'referer':'https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzYyNDkwNzI0Mg==&action=getalbum&album_id=4327161721590038539&subscene=159&subscene=&scenenote=https%3A%2F%2Fmp.weixin.qq.com%2Fs%2F4zYi45etCPP9-4RHwra6AQ&nolastread=1',
                        'cookie':'yyb_muid=0FB8311FCF4E61B010CB274ECE0D6033; RK=9Tk5OvZfNs; ptcz=d6e960b379d2656431578b872020f55b1014acee5e88ed20f65a0a324b86abff; qq_domain_video_guid_verify=393cfb67bef0c536; pgv_pvid=6784113444; _qimei_uuid42=1990f11180910093fcabf08b95967fa9a40d9e6655; _qimei_i_3=41f97683905d028ac9c4f8345e8425e7f6edaca346585280b08e280927932564636a65943c89e2a6bc8d; eas_sid=v137J6A1i2S9w6P5t372C3j2U6; pac_uid=0_pQac1eMdRkJHH; omgid=0_pQac1eMdRkJHH; _qimei_fingerprint=b2345e7399e16221ab8b1818665bf38a; _qimei_q36=; _qimei_h38=716de4a5fcabf08b95967fa90200000b61a51e; rewardsn=; wxtokenkey=777'}
        self.params = {
            'action' : 'getalbum',
            '__biz' : 'MzYyNDkwNzI0Mg ==',
            'album_id' : '4327161721590038539',
            'count' : '10',
            'begin_msgid' : '2247484038',
            'begin_itemidx' : '1',
            'uin' : '',
            'key' : '',
            'pass_ticket' : '',
            'wxtoken' : '',
            'devicetype' : '',
            'clientversion' : '',
            '__biz' : 'MzYyNDkwNzI0Mg%3D%3D',
            'appmsg_token' :'',
            'x5' : '0',
            'f' : 'json'}  #在网页上的负载里面直接扒下来的
        self.url_detail = []
        self.target_all = None         #接收后面的爬出来的八字
        self.text_all = None        #接收后面爬出来的评语
    def request_html_all(self):
        while 1:
            res = requests.get(self.url,headers=self.headers,params=self.params,timeout=12)
            res_json = res.json()
            msg_lst = res_json.get("getalbum_resp",[])    #如果没有就返回[],避免报错
            print(f'{msg_lst}\n')
            msg_art = msg_lst.get('article_list', [])
            if not msg_art:
                break
            if msg_art:                   #加一步避免报错,json太长了,看多了眼花,不管为什么了
                for item in msg_lst['article_list']:
                    self.url_detail.append(item['url'])           #把所有网址都装进这个self.url列表
                last_msg = msg_lst['article_list'][-1]
                msg_id = last_msg['msgid']
                self.params['begin_msgid'] = msg_id

            # print(res_json)
            # print('-'*23)
            # print(msg_lst)

    def save_html_all(self):
        with open('data/网页汇总.txt', mode='w', encoding='utf-8') as f:
            for i in self.url_detail:
                f.write(i)
                f.write('\n')

    def read_html_all(self):
        with open('data/网页汇总.txt',mode='r',encoding='utf-8')as f:
            count = 0
            for i in f:
                # count += 1
                # if count == 4:                          #先试试前三个
                #     break
                self.for_target(i)

    def for_target(self,x):
        res = requests.get(x,headers=self.headers,timeout=12)
        s = html.fromstring(res.content)
        text = s.xpath('//span[contains(@style,"font-weight")]/text()')
        # print(text)
        self.text_all = ''.join(text)
        try:
            target = re.search(r'([乾坤]{1})造{0,1}[:：]([\u4e00-\u9fa5]{2})[\s\xa0\u3000]{0,3}([\u4e00-\u9fa5]{2})[\s\xa0\u3000]{0,3}([\u4e00-\u9fa5]{2})[\s\xa0\u3000]{0,3}([\u4e00-\u9fa5]{2})',self.text_all)
        # print(repr(self.text_all))          #repr还原真实字符,防止有些空格符号特殊看不到
            self.target_all = target.group(1)+target.group(2)+target.group(3)+target.group(4)+target.group(5)
            # print(self.target_all)
            self.save_detail()      #调用下面保存数据
            time.sleep(0.5)
        except:
            print(repr(self.text_all))

    def save_detail(self):
        auto_in(self.target_all,self.text_all)

if __name__ == '__main__':
    res = Robot()
    # res.request_html_all()      #获取要爬的网址
    # res.save_html_all()         #保存爬的网址 与上面联动
    res.read_html_all()           #录入数据
