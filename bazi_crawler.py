# -*- coding: utf-8 -*-
"""
八字案例爬虫 —— 爬取公开博客/文章中的命理案例(八字 + 评语)
============================================================
输出与《八字自动录入数据》相同结构的 CSV:
  天干1,地支1,天干2,地支2,天干3,地支3,天干4,地支4,性别,合并,得分,评语
其中 得分 列填"未评分", 之后你可以用 bazi_scorer 的思路或人工来打分。

用法:
  python bazi_crawler.py                    # 爬取内置示例文章列表
  python bazi_crawler.py -u 网址1 网址2      # 爬取指定文章
  python bazi_crawler.py -f urls.txt        # 从文本文件读取网址(每行一个)
  python bazi_crawler.py --discover 博客主页  # 先抓页面上的文章链接再逐个爬
  python bazi_crawler.py -o 输出.csv        # 指定输出文件(默认 爬取案例数据.csv)

支持的站点(自动识别):
  * blog.sina.com.cn   新浪博客文章
  * www.360doc.com     360doc 个人图书馆
  * www.douban.com     豆瓣笔记

重要声明:
  * 仅用于爬取公开文章、个人学习研究, 勿用于商业用途;
  * 请遵守目标网站 robots 协议与使用条款, 本脚本已内置请求间隔(1~2.5秒)
    与重试机制, 请勿调快频率;
  * 请勿爬取受版权保护的整本图书、付费内容或需要登录的内容。
"""

import argparse
import csv
import os
import random
import re
import sys
import time
from html.parser import HTMLParser

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import requests

# ================= 常量 =================
GAN = '甲乙丙丁戊己庚辛壬癸'
ZHI = '子丑寅卯辰巳午未申酉戌亥'
GAN_MAP = {c: i + 1 for i, c in enumerate(GAN)}   # 与CSV编码一致: 1~10
ZHI_MAP = {c: i + 1 for i, c in enumerate(ZHI)}   # 1~12

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}
DELAY_MIN, DELAY_MAX = 2.0, 4.0   # 请求间隔(秒), 请保持礼貌

# 内置示例: 公开的命理案例文章(八字+反馈)。新浪桌面版有时反爬,
# 脚本会自动改用移动镜像 blog.sina.cn。
DEFAULT_URLS = [
    # --- 襄阳金笑易: 命例集系列 ---
    'https://blog.sina.com.cn/s/blog_9b6237b80101sn1b.html',   # 丧偶八字命例集(10例)
    'https://blog.sina.com.cn/s/blog_9b6237b80101gvex.html',   # 妓女小姐老鸨命例集(一)
    'https://blog.sina.com.cn/s/blog_9b6237b80101gvf9.html',   # 妓女小姐老鸨命例集(二)
    'https://blog.sina.com.cn/s/blog_9b6237b80101m6y3.html',   # 破财的八字命例集1
    'https://blog.sina.com.cn/s/blog_9b6237b80101l4mu.html',   # 亿万富翁八字命例集1
    'https://blog.sina.com.cn/s/blog_9b6237b80101h6nn.html',   # 小偷骗子的八字命例集(1)
    # --- 其他作者的公开案例文章 ---
    'https://blog.sina.com.cn/s/blog_1480887d801031s5y.html',  # 子平研究: 生死窍50例(上25例)
    'https://blog.sina.com.cn/s/blog_61c7481b0100ftqn.html',   # 论坛八字: 两例职业篇(含真实反馈)
    'https://blog.sina.com.cn/s/blog_61c7481b0100gn0l.html',   # 元亨利贞论坛: 简批实例集之十
    'https://blog.sina.com.cn/s/blog_659d6f2f0102x8c4.html',   # 命理: 建房耗财的八字预测
]

# 正文解析候选容器: 依次尝试, 取文本最长者
BODY_CANDIDATES = [
    {'ids': ['pl-blog-article'], 'classes': ['b-txt1']},                # 新浪博客移动版
    {'ids': ['sina_keyword_ad_area2'], 'classes': ['articalContent']},  # 新浪博客桌面版
    {'ids': ['art_content', 'content', 'articleContent']},              # 360doc(若网络可达)
    {'ids': ['link-report']},                                           # 豆瓣笔记
    {'tags': ['article']},                                              # 通用兜底
]

# 八字匹配: 乾/坤 (可选"造"/":") + 四柱, 柱间允许 ?、空格等分隔符
PAIR = '[%s][%s]' % (GAN, ZHI)
SEP = '[^%s%s]{0,4}' % (GAN, ZHI)
LEAD = '[^乾坤%s%s]{0,6}' % (GAN, ZHI)
CASE_RE = re.compile(
    '([乾坤])' + LEAD + '(' + PAIR + ')' + SEP + '(' + PAIR + ')' +
    SEP + '(' + PAIR + ')' + SEP + '(' + PAIR + ')')

# 竖排格式: "乾造：乙甲癸丙 / 卯申卯辰" —— 先4个天干, 再4个地支
CASE_RE_B = re.compile(
    '([乾坤])' + LEAD + '([%s]{4})' % GAN +
    '[^%s%s]{0,8}' % (GAN, ZHI) + '([%s]{4})' % ZHI)

# 空格分隔格式: "乾造：戊 戊 壬 癸 （从格） 午 午 戌 卯"
G_SEP = '[^%s%s]{1,4}' % (GAN, ZHI)
GG = '([%s])%s([%s])%s([%s])%s([%s])' % (GAN, G_SEP, GAN, G_SEP, GAN, G_SEP, GAN)
ZZ = '([%s])%s([%s])%s([%s])%s([%s])' % (ZHI, G_SEP, ZHI, G_SEP, ZHI, G_SEP, ZHI)
CASE_RE_C = re.compile('([乾坤])' + LEAD + GG + '[^%s%s]{0,30}' % (GAN, ZHI) + ZZ)

ARTICLE_LINK_RE = re.compile(
    r'href=["\']([^"\']*?(?:s/blog_\w+\.html|dpool/blog/s/blog_\w+\.html|content/\d+/\d+/\d+/\w+\.shtml|note/\d+/?)[^"\']*)["\']',
    re.I)


# ================= HTML 解析(标准库, 无第三方依赖) =================
class DivTextExtractor(HTMLParser):
    """提取指定 id/class 的 div 内文本(正确处理嵌套 div、br/p 换行)。"""

    def __init__(self, ids=(), classes=()):
        super().__init__(convert_charrefs=True)
        self.ids = set(ids)
        self.classes = set(classes)
        self.in_target = False
        self.target_depth = 0
        self.bufs = []
        self.cur = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'div':
            if self.in_target:
                self.target_depth += 1
                return
            hit_id = a.get('id') in self.ids
            hit_cls = any(c in self.classes for c in (a.get('class') or '').split())
            if hit_id or hit_cls:
                self.in_target = True
                self.target_depth = 1
                self.cur = []
                self.bufs.append(self.cur)
        elif self.in_target and tag in ('br', 'p', 'li', 'tr', 'h1', 'h2', 'h3', 'h4', 'blockquote'):
            self.cur.append('\n')

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_target:
            self.target_depth -= 1
            if self.target_depth == 0:
                self.in_target = False
                self.cur = None

    def handle_data(self, data):
        if self.in_target:
            self.cur.append(data)

    def texts(self):
        return [''.join(b).strip() for b in self.bufs]


def decode_bytes(content):
    """按常见中文编码依次尝试解码。"""
    for enc in ('utf-8', 'gb18030', 'gbk', 'big5'):
        try:
            return content.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return content.decode('utf-8', errors='replace')


def extract_body(html):
    """返回 (标题, 正文文本)。"""
    title = ''
    m = re.search(r'<title[^>]*>(.*?)</title>', html, re.S | re.I)
    if m:
        title = re.sub(r'\s+', ' ', m.group(1)).strip()
    best = ''
    for cand in BODY_CANDIDATES:
        p = DivTextExtractor(ids=cand.get('ids', ()), classes=cand.get('classes', ()))
        try:
            p.feed(html)
        except Exception:
            continue
        for t in p.texts():
            if len(t) > len(best):
                best = t
    if not best:
        # 最后兜底: 整个页面去标签
        best = re.sub(r'<script.*?</script>|<style.*?</style>', ' ', html, flags=re.S | re.I)
        best = re.sub(r'<[^>]+>', ' ', best)
        best = re.sub(r'\s+', ' ', best).strip()
    return title, best


# ================= 抓取 =================
def fetch(url, retries=3):
    for i in range(retries):
        try:
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            resp = requests.get(url, headers=HEADERS, timeout=25)
            if resp.status_code == 200 and resp.content:
                return decode_bytes(resp.content)
            print('  [{}] 状态码 {}'.format(resp.status_code, url))
        except Exception as e:
            print('  [第{}次失败] {}: {}'.format(i + 1, url, e))
    return None


# ================= 案例提取 =================
def extract_cases(text):
    """从正文里找出所有 乾/坤 + 四柱, 返回 [(性别, 4天干, 4地支, 标记起点), ...]。
    支持两种写法: 横排(癸丑 己未 甲子 辛未) 与 竖排(乙甲癸丙 / 卯申卯辰)。
    若同一八字重复出现只保留一次。"""
    matches = []
    for m in CASE_RE.finditer(text):
        pairs = [m.group(2), m.group(3), m.group(4), m.group(5)]
        matches.append((m.start(), m.group(1), pairs))
    for m in CASE_RE_B.finditer(text):
        gans4, zhis4 = m.group(2), m.group(3)
        pairs = [gans4[i] + zhis4[i] for i in range(4)]
        matches.append((m.start(), m.group(1), pairs))
    for m in CASE_RE_C.finditer(text):
        gans4 = [m.group(2), m.group(3), m.group(4), m.group(5)]
        zhis4 = [m.group(6), m.group(7), m.group(8), m.group(9)]
        pairs = [gans4[i] + zhis4[i] for i in range(4)]
        matches.append((m.start(), m.group(1), pairs))
    matches.sort()
    cases, seen = [], set()
    for start, gender, pairs in matches:
        key = gender + ''.join(pairs)
        if key in seen:
            continue
        seen.add(key)
        cases.append({'gender': gender,
                      'gans': [GAN_MAP[p[0]] for p in pairs],
                      'zhis': [ZHI_MAP[p[1]] for p in pairs],
                      'start': start, 'merge': key})
    return cases


def build_rows(cases, full_text, url, title):
    """按案例起止位置切分评语: 每个案例的评语 = 从它的标记到下一个案例标记之间的文字;
    单案例文章则用全文。"""
    rows = []
    n = len(cases)
    for i, c in enumerate(cases):
        if n == 1:
            comment = full_text
        else:
            end = cases[i + 1]['start'] if i + 1 < n else len(full_text)
            comment = full_text[c['start']:end].strip()
        rows.append({
            '天干1': c['gans'][0], '地支1': c['zhis'][0],
            '天干2': c['gans'][1], '地支2': c['zhis'][1],
            '天干3': c['gans'][2], '地支3': c['zhis'][2],
            '天干4': c['gans'][3], '地支4': c['zhis'][3],
            '性别': 1 if c['gender'] == '乾' else 2,
            '合并': c['merge'],
            '得分': '未评分',
            '评语': (title + '\n' + comment).strip(),
        })
    return rows


# ================= 链接发现 =================
def discover_links(html, base_url):
    """从列表页里找出文章链接(自动补全相对路径)。"""
    from urllib.parse import urljoin
    found = set()
    for l in ARTICLE_LINK_RE.findall(html):
        l = l.strip()
        if l.startswith('#'):
            continue
        found.add(urljoin(base_url, l))
    return sorted(found)


# ================= 主流程 =================
COLUMNS = ['天干1', '地支1', '天干2', '地支2', '天干3', '地支3',
           '天干4', '地支4', '性别', '合并', '得分', '评语']


def load_existing(path):
    if not os.path.exists(path):
        return []
    rows = []
    for enc in ('utf-8-sig', 'gbk'):
        try:
            with open(path, encoding=enc, newline='') as f:
                for r in csv.DictReader(f):
                    rows.append(r)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    return rows


def save_rows(path, rows):
    seen = set()
    for r in load_existing(path):
        seen.add(r.get('合并', ''))
    all_rows = load_existing(path)
    new = []
    for r in rows:                      # 本次爬取内部也按"合并"去重, 保留先出现的
        key = r.get('合并', '')
        if key in seen:
            continue
        seen.add(key)
        new.append(r)
    all_rows = all_rows + new
    if not all_rows:
        print('没有可保存的数据')
        return 0
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, '') for k in COLUMNS})
    return len(new)


def fetch_sina_with_fallback(url):
    """新浪博客: 桌面版可能反爬(418/空页面), 失败时自动改用移动镜像 blog.sina.cn。"""
    html = fetch(url, retries=1)
    if html is None or len(html) < 500:
        m = re.search(r'blog\.sina\.com\.cn/s/(blog_\w+\.html)$', url)
        if m:
            mirror = 'https://blog.sina.cn/dpool/blog/s/' + m.group(1)
            print('  [换移动镜像]', mirror)
            html = fetch(mirror)
    return html


def crawl_url(url, out_rows):
    if 'blog.sina.com.cn' in url:
        html = fetch_sina_with_fallback(url)
    else:
        html = fetch(url)
    if html is None:
        return 0
    title, body = extract_body(html)
    if not body:
        print('  [未提取到正文]', url)
        return 0
    cases = extract_cases(body)
    if not cases:
        print('  [未找到八字]', url, ' 标题:', title[:30])
        return 0
    rows = build_rows(cases, body, url, title)
    out_rows.extend(rows)
    print('  [OK] {}  提取 {} 个八字'.format(url, len(cases)))
    for r in rows:
        print('       ', r['合并'])
    return len(cases)


def main():
    ap = argparse.ArgumentParser(description='爬取公开博客中的八字案例(八字+评语)')
    ap.add_argument('-u', '--urls', nargs='+', help='要爬取的文章网址(可多个)')
    ap.add_argument('-f', '--file', help='从文本文件读取网址列表(每行一个)')
    ap.add_argument('--discover', help='先抓取该页面上的文章链接, 再逐个爬取')
    ap.add_argument('--max-pages', type=int, default=30, help='--discover 时最多爬多少篇文章')
    ap.add_argument('-o', '--output', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '爬取案例数据.csv'))
    args = ap.parse_args()

    urls = []
    if args.urls:
        urls.extend(args.urls)
    if args.file:
        with open(args.file, encoding='utf-8-sig', errors='replace') as f:
            urls.extend(line.strip() for line in f if line.strip() and not line.startswith('#'))
    if args.discover:
        html = fetch(args.discover)
        if html:
            found = discover_links(html, args.discover)
            print('从 {} 发现 {} 个文章链接'.format(args.discover, len(found)))
            urls.extend(found)
    if not urls:
        urls = DEFAULT_URLS
    if args.discover:
        urls = urls[:args.max_pages]

    print('共 {} 个网址待爬取'.format(len(urls)))
    out_rows = []
    ok, total_cases = 0, 0
    for u in urls:
        try:
            n = crawl_url(u, out_rows)
            if n:
                ok += 1
                total_cases += n
        except Exception as e:
            print('  [异常]', u, e)
    print('-' * 60)
    print('成功 {} 个页面, 提取八字 {} 个'.format(ok, total_cases))
    added = save_rows(args.output, out_rows)
    print('本次新增 {} 条, 保存到: {}'.format(added, args.output))
    print('总数据量: {} 条'.format(len(load_existing(args.output))))


if __name__ == '__main__':
    main()
