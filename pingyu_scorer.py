# -*- coding: utf-8 -*-
"""
评语自动评分器 —— 神经网络(TextCNN) + 评分规则 融合
====================================================
按之前人工评分的逻辑(0~4)对"评语"文本打分:
  4 = 大富大贵(亿万/厅级以上高官, 人生总体完整)
  3 = 中上层次(千万/处级/名流; 或大富但横死早亡)
  2 = 普通中等(温饱小康、普通公职白领; 婚灾破财等不顺)
  1 = 困苦(贫困/残障/精神病/牢狱/风尘/社会底层)
  0 = 身亡或人生毁灭(横死/病亡/早夭/绝症)

两个模块:
  1. 神经网络: 字符级 TextCNN, 在《八字自动录入数据(AI评分).csv》的155条
     标注数据上训练(带类别加权、Dropout、早停, 并报告验证集基线对比);
  2. 规则打分: 把上述评分逻辑写成关键词规则(如"亿万/厅级"->4, "牢狱/贫困"->1,
     "去世/病逝"->0, "破产/负债"削弱财富分...);
  最终得分 = 规则置信度 × 规则档 + (1-置信度) × 神经网络概率, 取最大概率档。

用法:
  python pingyu_scorer.py train                      # 训练神经网络并保存模型
  python pingyu_scorer.py score 爬取案例数据.csv      # 对CSV的"评语"列逐行打分,
                                                     # 输出 文件名(AI评分).csv
  python pingyu_scorer.py score xxx.csv -o 结果.csv  # 指定输出文件
  python pingyu_scorer.py text "千万富翁，二婚..."     # 直接给一段评语文本打分

说明:
  * 训练数据只有155条, 神经网络部分泛化能力有限(验证集准确率约与"全猜2分"
    基线持平), 因此规则部分承担主要信号, 神经网络负责平滑与兜底;
  * 规则是评分逻辑的近似实现, 对复杂案例(如"亿万富翁触电身亡"这类)可能与
    人工判断相差一档, 属正常;
  * 分数仅供参考, 建议人工抽查修正后再用于训练八字模型。
"""

import argparse
import csv
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ================= 配置 =================
LABELED_DATA = r'D:\数据分析\信仰科学\data\八字自动录入数据(AI评分).csv'   # 训练用标注数据
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pingyu_scorer.pt')

SEED = 42
MAX_LEN = 1200          # 每条评语最多取前1200字
EMB_DIM = 64
KERNELS = (3, 4, 5)     # TextCNN 卷积核宽度
N_FILTERS = 64
HIDDEN = 64
DROPOUT = 0.5
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 15
N_CLASSES = 5

CHARS = ''


# ================= 规则部分(评分逻辑的关键词实现) =================
DEATH = ['十死无生', '英年早逝', '早夭', '夭亡', '夭折', '夭命', '不禄', '寿尽',
         '去世', '病逝', '离世', '死亡', '过世', '亡故', '身故', '逝世',
         '遇害', '被杀', '刀杀', '身亡', '丧命', '毙命', '遇难', '至死', '致死',
         '死于', '自杀', '枪决', '抢救无效', '停止跳动']
WEALTH4 = ['亿万', '十亿', '上亿', '正厅', '副部', '部级', '厅长', '高院院长',
           '市委书记', '省部级', '厅级']
WEALTH3 = ['千万', '五千万', '几千万', '三千万', '两千万', '处级', '处长',
           '组织部长', '明星', '富翁', '最红', '大火', '爆红', '走红', '名流']
NEGATE = ['破产', '负债', '清零', '烧完', '血本无归', '破财', '欠债', '赔光',
          '败的一塌糊涂', '一无所有']
STRONG_DISASTER = ['精神病', '牢狱', '入狱', '坐牢', '判刑', '无期', '死缓',
                   '风尘', '小姐', '下海', '卖淫', '妓女', '老鸨', '陪酒',
                   '贫困', '贫穷', '穷人', '底层', '低保', '五保', '弃婴',
                   '抛弃', '残疾', '残废', '失明', '眼盲', '盲师', '脑瘫',
                   '石女', '无子宫', '帕金森', '植物人', '瘫痪', '杀人',
                   '分尸', '涉黑', '黑社会', '毒品', '电诈', '诈骗', '小偷',
                   '偷盗', '被强暴', '强暴', '强奸']
WEAK_DISASTER = ['白血病', '血癌', '癌症', '胃癌', '肝癌', '子宫癌', '肺癌',
                 '贫血', '抑郁', '抑郁症']

SUBJECT_ROLE = ['命主', '本人', '此女', '此男', '自己']
FAMILY_ROLE = ['妻子', '丈夫', '老公', '老婆', '配偶', '父亲', '母亲', '父母',
               '妻', '夫', '父', '母', '儿子', '女儿', '孩子', '小孩', '哥',
               '弟', '姐', '妹', '兄', '爷爷', '奶奶', '姥姥', '姥爷', '岳父',
               '岳母', '公公', '婆婆', '长子', '长女', '次子', '三女儿']
THIRD_ROLE = ['客户', '朋友', '同学', '邻居', '同事', '路人', '乘客', '司机',
              '对方', '别人', '他人']
NEG_DEATH_RE = re.compile(r'(不|未|没|无|非)(会|曾|有|能)?(死亡|去世|病逝|离世|过世|亡故|身故|逝世|遇害|被杀|身亡|丧命|毙命|遇难|自杀)')


def _nearest_role(win):
    """在死亡词前12字的窗口里找最近的角色词, 返回 'subject'/'family'/'third'/None。"""
    best_pos, best_kind = -1, None
    for kind, roles in (('subject', SUBJECT_ROLE), ('family', FAMILY_ROLE),
                        ('third', THIRD_ROLE)):
        pos = -1
        for r in roles:
            p = win.rfind(r)
            if p > pos:
                pos = p
        if pos > best_pos:
            best_pos, best_kind = pos, kind
    return best_kind


def _classify_deaths(text):
    """把文本里的死亡词分类: 返回 (命主本人死亡?, 家人/他人死亡?)。"""
    text2 = NEG_DEATH_RE.sub('', text)   # 去掉"不会死亡"这类否定表述
    n_subject = n_other = n_bare = 0
    for m in re.finditer('|'.join(sorted(DEATH, key=len, reverse=True)), text2):
        win = text2[max(0, m.start() - 20):m.start()]
        kind = _nearest_role(win)
        if kind == 'subject':
            n_subject += 1
        elif kind in ('family', 'third'):
            n_other += 1
        else:
            n_bare += 1
    if n_subject > 0:
        return True, False                # 命主本人死亡
    if n_other > 0 and n_bare == 0:
        return False, True                # 只有家人/他人死亡
    if n_bare > 0 and n_other == 0:
        return True, False                # 无角色词, 视作本人死亡
    return False, bool(n_other)           # 混杂, 保守处理为家人/他人死亡


def rule_score(text):
    """按评分逻辑给文本打分, 返回 (分数, 置信度, 命中词说明)。"""
    subject_death, family_death = _classify_deaths(text)
    has_w4 = [w for w in WEALTH4 if w in text]
    has_w3 = [w for w in WEALTH3 if w in text]
    has_neg = [w for w in NEGATE if w in text]
    has_sd = [w for w in STRONG_DISASTER if w in text]
    has_wd = [w for w in WEAK_DISASTER if w in text]
    reason = []
    if subject_death:
        reason.append('命主死亡')
    if family_death:
        reason.append('家人/他人死亡')
    if has_w4:
        reason.append('大富词:' + ','.join(has_w4[:3]))
    if has_w3:
        reason.append('中上词:' + ','.join(has_w3[:3]))
    if has_neg:
        reason.append('破败词:' + ','.join(has_neg[:3]))
    if has_sd:
        reason.append('灾祸词:' + ','.join(has_sd[:3]))
    if has_wd:
        reason.append('病灾词:' + ','.join(has_wd[:3]))

    if subject_death and (has_w4 or has_w3):
        return 3, 0.9, reason          # 大富大贵但身亡 -> 3
    if subject_death:
        return 0, 0.9, reason          # 命主身亡 -> 0
    if has_w4 and not has_neg:
        return 4, 0.9, reason          # 亿万/厅级 -> 4
    if has_w3 and not has_neg:
        return 3, 0.8, reason          # 千万/名流 -> 3
    if has_sd:
        return 1, 0.8, reason          # 灾祸困苦 -> 1
    if family_death:
        return 2, 0.7, reason          # 丧偶/家人亡 -> 2
    if has_wd:
        return 1, 0.7, reason          # 本人重病 -> 1
    if has_w4 or has_w3:
        return 2, 0.7, reason          # 曾大富但破败归零 -> 2
    return 2, 0.3, reason              # 默认2分, 弱置信


# ================= 文本预处理 =================
def clean_text(s):
    s = re.sub(r'https?://\S+', ' ', s)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def build_vocab(texts, min_count=2):
    from collections import Counter
    cnt = Counter()
    for t in texts:
        cnt.update(t)
    vocab = ['<PAD>', '<UNK>']
    for ch, c in cnt.most_common():
        if c >= min_count and ch not in vocab:
            vocab.append(ch)
    vmap = {ch: i for i, ch in enumerate(vocab)}
    return vmap, vocab


def encode_text(s, vmap, max_len=MAX_LEN):
    s = clean_text(s)[:max_len]
    ids = [vmap.get(ch, 1) for ch in s]
    return ids


def pad(ids, max_len=MAX_LEN):
    ids = ids[:max_len]
    return ids + [0] * (max_len - len(ids))


class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.x = [pad(encode_text(t, VMAP)) for t in texts]
        self.y = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = torch.tensor(self.x[i], dtype=torch.long)
        if self.y is not None:
            return x, torch.tensor(self.y[i], dtype=torch.long)
        return x


# ================= 模型 =================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=EMB_DIM, kernels=KERNELS,
                 n_filters=N_FILTERS, hidden=HIDDEN, dropout=DROPOUT,
                 n_classes=N_CLASSES):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in kernels])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(n_filters * len(kernels), hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes))

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)          # (B, emb, L)
        feats = [torch.max(torch.relu(c(e)), dim=2)[0] for c in self.convs]
        h = torch.cat(feats, dim=1)
        return self.fc(self.drop(h))


# ================= 加载标注数据 =================
def load_labeled():
    df = None
    for enc in ('utf-8-sig', 'gbk'):
        try:
            df = pd.read_csv(LABELED_DATA, encoding=enc, dtype=str)
            break
        except (UnicodeDecodeError, UnicodeError, FileNotFoundError):
            continue
    if df is None:
        raise FileNotFoundError('找不到标注数据: ' + LABELED_DATA)
    target = 'AI评分' if 'AI评分' in df.columns else '得分'
    if target == '得分':
        df = df[df['得分'] != '未评分'].reset_index(drop=True)
    texts = df['评语'].fillna('').apply(clean_text).tolist()
    y = df[target].astype(int).values
    return texts, y


def stratified_split(y, val_frac=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    tr_idx, va_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        va_idx.extend(idx[:n_val])
        tr_idx.extend(idx[n_val:])
    return np.array(tr_idx), np.array(va_idx)


# ================= 训练 =================
def train():
    texts, y = load_labeled()
    global VMAP, VOCAB
    VMAP, VOCAB = build_vocab(texts)
    print('标注数据 {} 条, 词表大小 {}'.format(len(y), len(VOCAB)))
    print('标签分布:', {int(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()})
    maj = int(np.argmax(np.bincount(y)))
    print('基线(全猜{}分)准确率: {:.3f}'.format(maj, (y == maj).mean()))
    print('-' * 60)

    accs, f1s = [], []
    best_state = None
    for run in range(1, 4):
        torch.manual_seed(SEED + run)
        np.random.seed(SEED + run)
        tr_idx, va_idx = stratified_split(y, seed=SEED + run)
        tr_ds = TextDataset([texts[i] for i in tr_idx], y[tr_idx])
        va_ds = TextDataset([texts[i] for i in va_idx], y[va_idx])
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

        counts = np.bincount(y[tr_idx], minlength=N_CLASSES).astype(float)
        counts[counts == 0] = 1
        w = np.sqrt(counts.sum() / (N_CLASSES * counts))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
        model = TextCNN(len(VOCAB))
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_loss, bad, run_acc, run_f1, run_state = float('inf'), 0, 0, 0, None
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for x, lab in tr_loader:
                opt.zero_grad()
                loss = loss_fn(model(x), lab)
                loss.backward()
                opt.step()
            model.eval()
            vloss, preds, trues = 0.0, [], []
            with torch.no_grad():
                for x, lab in va_loader:
                    out = model(x)
                    vloss += loss_fn(out, lab).item() * len(lab)
                    preds.extend(out.argmax(1).tolist())
                    trues.extend(lab.tolist())
            vloss /= len(va_ds)
            if vloss < best_loss:
                best_loss = vloss
                bad = 0
                preds = np.array(preds)
                trues = np.array(trues)
                run_acc = float((preds == trues).mean())
                f1s_ = []
                for c in range(N_CLASSES):
                    tp = ((preds == c) & (trues == c)).sum()
                    if tp:
                        p_ = tp / (preds == c).sum()
                        r_ = tp / (trues == c).sum()
                        f1s_.append(2 * p_ * r_ / (p_ + r_))
                run_f1 = float(np.mean(f1s_)) if f1s_ else 0.0
                run_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        accs.append(run_acc)
        f1s.append(run_f1)
        print('第{}次: 验证集准确率 {:.3f}  macro-F1 {:.3f}'.format(run, run_acc, run_f1))
    print('-' * 60)
    print('神经网络平均: 准确率 {:.3f}±{:.3f}  macro-F1 {:.3f}±{:.3f}'.format(
        np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)))
    print('注: 与基线(全猜2分)持平属正常, 数据量太小; 规则部分承担主要信号。')

    # 用全部数据训练最终模型
    print('用全部 {} 条数据训练最终模型...'.format(len(y)))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    full_ds = TextDataset(texts, y)
    loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)
    model = TextCNN(len(VOCAB))
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    counts[counts == 0] = 1
    w = np.sqrt(counts.sum() / (N_CLASSES * counts))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(60):
        model.train()
        for x, lab in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), lab)
            loss.backward()
            opt.step()
    torch.save({'state_dict': model.state_dict(), 'vocab': VOCAB}, MODEL_PATH)
    print('模型已保存:', MODEL_PATH)


# ================= 预测 =================
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError('模型不存在, 请先运行: python pingyu_scorer.py train')
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    global VOCAB, VMAP
    VOCAB = ckpt['vocab']
    VMAP = {ch: i for i, ch in enumerate(VOCAB)}
    model = TextCNN(len(VOCAB))
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def predict_text(model, text):
    """融合规则与神经网络, 返回 (最终分, 各档概率, 规则分, 规则说明)。"""
    r, conf, reason = rule_score(text)
    x = torch.tensor([pad(encode_text(text, VMAP))], dtype=torch.long)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()
    onehot = np.zeros(N_CLASSES)
    onehot[r] = 1.0
    blend = conf * onehot + (1 - conf) * probs
    return int(blend.argmax()), blend, r, reason


def score_csv(path, out):
    model = load_model()
    df = None
    for enc in ('utf-8-sig', 'gbk'):
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str)
            break
        except (UnicodeDecodeError, UnicodeError, FileNotFoundError):
            continue
    if df is None:
        raise FileNotFoundError('找不到CSV: ' + path)
    if '评语' not in df.columns:
        raise ValueError('CSV 中没有"评语"列: ' + path)

    scores, rules, reasons = [], [], []
    for t in df['评语'].fillna(''):
        s, _, r, reason = predict_text(model, t)
        scores.append(s)
        rules.append(r)
        reasons.append(';'.join(reason))
    df['AI评分'] = scores
    df['规则分'] = rules
    df['评分依据'] = reasons
    if not out:
        base, ext = os.path.splitext(path)
        out = base + '(AI评分)' + ext
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print('完成: {} 行 -> {}'.format(len(df), out))
    print('分数分布:', {int(k): int(v) for k, v in pd.Series(scores).value_counts().sort_index().items()})
    return out


def main():
    ap = argparse.ArgumentParser(description='评语自动评分器(神经网络+规则)')
    ap.add_argument('cmd', choices=['train', 'score', 'text'])
    ap.add_argument('target', nargs='?', help='score: CSV路径; text: 评语文本')
    ap.add_argument('-o', '--output', default='', help='score时指定输出文件')
    args = ap.parse_args()

    if args.cmd == 'train':
        train()
    elif args.cmd == 'score':
        if not args.target:
            print('请提供CSV路径: python pingyu_scorer.py score xxx.csv')
            sys.exit(1)
        score_csv(args.target, args.output)
    elif args.cmd == 'text':
        if not args.target:
            print('请提供评语文本: python pingyu_scorer.py text "千万富翁..."')
            sys.exit(1)
        model = load_model()
        s, blend, r, reason = predict_text(model, args.target)
        print('规则分: {}  依据: {}'.format(r, ';'.join(reason) if reason else '无命中词, 默认2分'))
        for i, p in enumerate(blend):
            print('  {}分: {:.1%}  {}'.format(i, p, '#' * int(p * 40)))
        print('最终评分: {} 分'.format(s))


if __name__ == '__main__':
    main()
