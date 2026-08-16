# -*- coding: utf-8 -*-
"""
八字自动评分 —— 深度学习神经网络（PyTorch）
================================================
功能:
  1. 读取《八字自动录入数据(AI评分).csv》
  2. 用 4 柱干支 + 性别 作为特征, AI评分(0~4) 作为标签
  3. 训练一个带 Embedding 的小型神经网络(五分类)
  4. 保存模型, 之后输入八字即可自动评分

用法:
  python bazi_scorer.py                          # 无模型时先训练, 再进入交互式预测
  python bazi_scorer.py train                    # 训练模型(多次评估 + 最终全量训练并保存)
  python bazi_scorer.py evaluate                 # 在验证集上评估(多次运行取平均)
  python bazi_scorer.py predict 乾癸丑己未甲子辛未   # 预测一个八字
  python bazi_scorer.py predict 癸丑 己未 甲子 辛未 乾  # 带空格写法也支持

输入格式:
  前缀 乾/坤 表示性别(男/女), 后面 8 个字为 天干地支 交替(年柱月柱日柱时柱),
  例如: 乾癸丑己未甲子辛未
  不加前缀时默认按 乾(男) 处理, 也可在末尾写 1(男)/2(女)。

重要说明(请先读):
  * 数据只有 155 条, 且分数分布不均(2分占52%), 模型在验证集上的准确率
    与"全猜2分"的基线基本持平 —— 换句话说, 这份数据量不足以让神经网络
    学到可靠的"八字->分数"规律, 只能学到统计层面的微弱倾向;
  * 分数本身来自对"评语"的主观评分, 不是八字的数学函数, 模型学的是
    这份标注的口径;
  * 训练脚本已做 分层划分验证集、类别加权、早停、Dropout/权重衰减、
    多次评估取平均 等正规防过拟合措施, 输出会同时给出基线准确率,
    方便你判断模型到底有没有真学到东西;
  * 想真正提升效果的正确方向: ① 增加带标注的样本量(几百上千条);
    ② 把"评语"文本也喂给模型(文本分类/NLP), 而不仅仅用8个字;
    ③ 特征上加入五行强弱、十神、冲合刑害等命理特征。
"""

import os
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
# 数据文件路径(按需修改)
DATA_PATH = r'D:\数据分析\信仰科学\data\八字自动录入数据(AI评分).csv'
# 模型保存位置(默认与脚本同目录)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bazi_scorer.pt')

SEED = 42
N_RUNS = 5            # 评估时重复训练次数(取平均, 降低随机性)
VAL_FRAC = 0.2        # 验证集比例
EPOCHS = 400          # 单次训练最大轮数
PATIENCE = 50         # 早停耐心(验证损失不降多少轮就停)
EPOCHS_FINAL = 300    # 最终全量训练轮数
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 2e-3
EMB_DIM = 8           # 干支组合(柱)的 Embedding 维度
HIDDEN = [32]         # 全连接隐层(刻意小, 防过拟合)
DROPOUT = 0.6         # Dropout 比例
N_CLASSES = 5         # 0~4 五个分数档

# 天干、地支字表(与 CSV 中 1~10、1~12 的编码一致)
TIANGAN = '甲乙丙丁戊己庚辛壬癸'
DIZHI = '子丑寅卯辰巳午未申酉戌亥'
GAN_MAP = {c: i for i, c in enumerate(TIANGAN)}
ZHI_MAP = {c: i for i, c in enumerate(DIZHI)}


# ================= 数据 =================
def load_data():
    """读取 CSV, 返回 (天干, 地支, 性别, 标签) 及原始 DataFrame。"""
    df = None
    for enc in ('utf-8-sig', 'gbk'):
        try:
            df = pd.read_csv(DATA_PATH, encoding=enc, dtype=str)
            break
        except (UnicodeDecodeError, UnicodeError, FileNotFoundError):
            continue
    if df is None:
        raise FileNotFoundError('找不到数据文件: ' + DATA_PATH)

    # 优先用 AI评分 列; 没有的话退回 得分 列并去掉"未评分"的行
    if 'AI评分' in df.columns:
        target = 'AI评分'
    elif '得分' in df.columns:
        target = '得分'
        df = df[df['得分'] != '未评分'].reset_index(drop=True)
    else:
        raise ValueError('CSV 中既没有 AI评分 列也没有 得分 列')

    gan = df[['天干1', '天干2', '天干3', '天干4']].astype(int).values - 1   # 0~9
    zhi = df[['地支1', '地支2', '地支3', '地支4']].astype(int).values - 1   # 0~11
    sex = df['性别'].astype(int).values - 1                                # 0=乾(男) 1=坤(女)
    y = df[target].astype(int).values
    return gan, zhi, sex, y, df, target


def make_pillar(gan, zhi):
    """干支组合索引: 天干*12 + 地支, 范围 0~119。"""
    return gan * 12 + zhi


class BaZiDataset(Dataset):
    def __init__(self, pillar, sex, y):
        self.pillar = torch.tensor(pillar, dtype=torch.long)
        self.sex = torch.tensor(sex, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.pillar[i], self.sex[i], self.y[i]


def stratified_split(y, val_frac=VAL_FRAC, seed=SEED):
    """按类别分层划分训练/验证集。"""
    rng = np.random.default_rng(seed)
    tr_idx, va_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        va_idx.extend(idx[:n_val])
        tr_idx.extend(idx[n_val:])
    return np.array(tr_idx), np.array(va_idx)


def class_weights(y_train):
    """类别加权(开方软化): 样本少的类别权重略高, 又不过分追逐少数类。"""
    counts = np.bincount(y_train, minlength=N_CLASSES).astype(float)
    counts[counts == 0] = 1.0
    w = np.sqrt(counts.sum() / (N_CLASSES * counts))
    return torch.tensor(w, dtype=torch.float32)


# ================= 模型 =================
class BaZiNet(nn.Module):
    """4 柱干支各自 Embedding 后拼接, 再接小 MLP, 输出 5 个分数档概率。"""

    def __init__(self, emb_dim=EMB_DIM, hidden=HIDDEN, dropout=DROPOUT, n_classes=N_CLASSES):
        super().__init__()
        self.pillar_emb = nn.Embedding(120, emb_dim)
        in_dim = 4 * emb_dim + 1   # 4柱向量 + 性别
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)

    def forward(self, pillar, sex):
        p = self.pillar_emb(pillar).flatten(1)
        s = sex.unsqueeze(1).float()
        x = torch.cat([p, s], dim=1)
        return self.head(self.mlp(x))


# ================= 训练 / 评估 =================
def compute_metrics(y_true, y_pred):
    """准确率、宏平均 F1、以及每个类别的 P/R/F1/样本数。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = sorted(set(y_true.tolist()))
    per = {}
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per[c] = (p, r, f1, int((y_true == c).sum()))
    acc = float((y_pred == y_true).mean())
    macro_f1 = float(np.mean([per[c][2] for c in classes]))
    return acc, macro_f1, per


def train_one_split(pillar, sex, y, seed, verbose=False):
    """用固定随机种子训练一次, 返回 (验证集acc, macro-F1, 各类指标, 最佳轮数)。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_idx, va_idx = stratified_split(y, VAL_FRAC, seed)
    train_ds = BaZiDataset(pillar[tr_idx], sex[tr_idx], y[tr_idx])
    val_ds = BaZiDataset(pillar[va_idx], sex[va_idx], y[va_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BaZiNet()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y[tr_idx]))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')
    best_state = None
    best_epoch = 0
    bad = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for p, s, lab in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(p, s), lab)
            loss.backward()
            opt.step()
        model.eval()
        val_loss, preds, trues = 0.0, [], []
        with torch.no_grad():
            for p, s, lab in val_loader:
                out = model(p, s)
                val_loss += loss_fn(out, lab).item() * len(lab)
                preds.extend(out.argmax(1).tolist())
                trues.extend(lab.tolist())
        val_loss /= len(val_ds)
        sched.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    acc, macro_f1, per = compute_metrics(np.array(trues), np.array(preds))
    return acc, macro_f1, per, best_epoch


def train():
    gan, zhi, sex, y, df, target = load_data()
    pillar = make_pillar(gan, zhi)
    print('=' * 60)
    print('数据:', DATA_PATH)
    print('样本数:', len(y), ' 标签列:', target)
    print('标签分布:', {int(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()})
    maj = int(np.argmax(np.bincount(y)))
    print('基线(全猜{}分)准确率: {:.3f}'.format(maj, (y == maj).mean()))
    print('=' * 60)

    # 多次训练评估, 给出更可信的验证集成绩
    accs, f1s = [], []
    last_per = None
    for run in range(1, N_RUNS + 1):
        acc, f1, per, best_epoch = train_one_split(pillar, sex, y, seed=SEED + run - 1)
        accs.append(acc)
        f1s.append(f1)
        last_per = per
        print('第{}次训练: 验证集准确率 {:.3f}  macro-F1 {:.3f}  (最佳轮数 {})'.format(run, acc, f1, best_epoch))
    print('-' * 60)
    print('平均: 验证集准确率 {:.3f} ± {:.3f}   macro-F1 {:.3f} ± {:.3f}'.format(
        np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)))
    if last_per is not None:
        print('最后一次训练的各分数档指标(验证集):')
        for c in sorted(last_per):
            p, r, f1, n = last_per[c]
            print('  {}分: 精确率 {:.2f}  召回率 {:.2f}  F1 {:.2f}  样本 {}'.format(c, p, r, f1, n))
    print('=' * 60)
    print('注: 若准确率没有明显超过基线, 说明数据量不足以支撑可靠的泛化,')
    print('    模型只能当作学习示例和"统计倾向参考", 不能当作可靠的评分器。')

    # 用全部数据训练最终模型并保存
    print('正在用全部 {} 条数据训练最终模型...'.format(len(y)))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    full_ds = BaZiDataset(pillar, sex, y)
    loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)
    model = BaZiNet()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(1, EPOCHS_FINAL + 1):
        model.train()
        for p, s, lab in loader:
            opt.zero_grad()
            loss = loss_fn(model(p, s), lab)
            loss.backward()
            opt.step()
    save_model(model)
    print('模型已保存到:', MODEL_PATH)
    print('现在可以运行: python bazi_scorer.py predict 乾癸丑己未甲子辛未')


def evaluate():
    gan, zhi, sex, y, df, target = load_data()
    pillar = make_pillar(gan, zhi)
    maj = int(np.argmax(np.bincount(y)))
    print('在验证集上重复评估 {} 次...'.format(N_RUNS))
    accs, f1s = [], []
    for run in range(1, N_RUNS + 1):
        acc, f1, _, _ = train_one_split(pillar, sex, y, seed=SEED + run - 1)
        accs.append(acc)
        f1s.append(f1)
        print('第{}次: 验证集准确率 {:.3f}  macro-F1 {:.3f}'.format(run, acc, f1))
    print('平均: 准确率 {:.3f} ± {:.3f}   macro-F1 {:.3f} ± {:.3f}'.format(
        np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)))
    print('基线(全猜{}分): 准确率 {:.3f}'.format(maj, (y == maj).mean()))


# ================= 保存 / 加载 / 预测 =================
def save_model(model):
    torch.save({
        'state_dict': model.state_dict(),
        'emb_dim': EMB_DIM,
        'hidden': HIDDEN,
        'dropout': DROPOUT,
        'n_classes': N_CLASSES,
    }, MODEL_PATH)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError('模型不存在, 请先运行: python bazi_scorer.py train')
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    model = BaZiNet(emb_dim=ckpt['emb_dim'], hidden=ckpt['hidden'],
                    dropout=ckpt['dropout'], n_classes=ckpt['n_classes'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def parse_input(s):
    """把用户输入的八字字符串解析成 (天干索引[4], 地支索引[4], 性别0/1)。"""
    s = s.strip()
    if not s:
        raise ValueError('输入为空')
    gender = None
    # 前缀 乾/坤
    if s[0] in '乾坤':
        gender = 0 if s[0] == '乾' else 1
        s = s[1:]
    # 去掉所有空白
    s = ''.join(ch for ch in s if not ch.isspace())
    # 末尾数字 1/2 表示性别
    if s[-1:] in ('1', '2'):
        gender = int(s[-1]) - 1
        s = s[:-1]
    # 只保留天干地支字符
    s = ''.join(ch for ch in s if ch in GAN_MAP or ch in ZHI_MAP)
    if len(s) != 8:
        raise ValueError('天干地支应为 8 个字(如 癸丑己未甲子辛未), 实际解析出 {} 个字: {}'.format(len(s), s))
    gan = [GAN_MAP[s[i]] for i in range(0, 8, 2)]
    zhi = [ZHI_MAP[s[i]] for i in range(1, 8, 2)]
    if gender is None:
        gender = 0
        print('[提示] 未指定性别, 默认按 乾(男) 处理; 可加前缀 乾/坤 或末尾 1/2')
    return gan, zhi, gender


def predict_one(model, s):
    gan, zhi, sex = parse_input(s)
    pillar = np.array(gan) * 12 + np.array(zhi)
    p = torch.tensor(np.stack([pillar]), dtype=torch.long)
    sx = torch.tensor([sex], dtype=torch.long)
    with torch.no_grad():
        logits = model(p, sx)
        probs = torch.softmax(logits, dim=1)[0].numpy()
    score = int(probs.argmax())
    return score, probs, sex


def predict(s):
    model = load_model()
    score, probs, sex = predict_one(model, s)
    print('八字: {}  性别: {}'.format(s, '乾(男)' if sex == 0 else '坤(女)'))
    for i, p in enumerate(probs):
        bar = '#' * int(round(p * 40))
        print('  {}分: {:.2%}  {}'.format(i, p, bar))
    print('预测评分: {} 分'.format(score))
    return score


def interactive():
    if not os.path.exists(MODEL_PATH):
        print('未找到已训练模型, 先开始训练...')
        train()
    model = load_model()
    print('=' * 60)
    print('交互式八字评分(输入 乾癸丑己未甲子辛未 这样的格式)')
    print('输入 exit 或直接回车退出')
    print('=' * 60)
    while True:
        try:
            s = input('请输入八字: ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not s or s.lower() == 'exit':
            break
        try:
            score, probs, sex = predict_one(model, s)
            print(' -> 预测评分: {} 分'.format(score))
            top = int(probs.argmax())
            print('    (最可能 {} 分, 概率 {:.1%}; 各档概率: {})'.format(
                top, probs[top],
                ', '.join('{}分{:.0%}'.format(i, p)for i, p in enumerate(probs))))
        except ValueError as e:
            print('输入有误:', e)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        interactive()
    elif args[0] == 'train':
        train()
    elif args[0] == 'evaluate':
        evaluate()
    elif args[0] == 'predict':
        if len(args) > 1:
            predict(' '.join(args[1:]))
        else:
            s = input('请输入八字(如 乾癸丑己未甲子辛未): ')
            predict(s)
    else:
        print(__doc__)
