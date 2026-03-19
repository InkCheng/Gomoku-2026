# UTSS 在线预训练脚本（动态生成样本 + 7通道 + 轮流采样）===
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
import pickle, ast
import numpy as np
from tqdm import tqdm
import gomoku_engine                # C++ 封装的五子棋逻辑模块
from src.models.policy_value_utss_net import Net  # 三头神经网络结构（策略/价值/UTSS）

# ---------- 参数配置 ----------
BOARD = 15               # 棋盘大小：15×15
PLANES = 7               # 7个输入通道
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH = 512              # 每个 batch 样本数
EPOCH = 3                # 训练轮数
MAX_STEPS = 10000        # 最多训练步数（控制总样本数）
hash_paths = {
    0: "merged_hash_new.pkl",       # label 0：必胜局面
    2: "lossmerged_hash_new.pkl",   # label 2：必败局面
    1: "checkmerged_hash_new.pkl",  # label 1：平局
    3: "limitedmerged_hash_new.pkl" # label 3：超时/未知
}

# 读取哈希表
def load_hash_table(path):
    d = {}
    with open(path, "rb") as f:
        while True:
            try:
                d.update(pickle.load(f))  # 多个 dict 累加合并
            except EOFError:
                break
    return d

# 解析字符串棋盘
def deserialize_board(list_str: str):
    rows = ast.literal_eval(list_str)     # 将字符串还原为 list[str]
    flat = ''.join(rows)                  # 合并为225个字符
    assert len(flat) == BOARD * BOARD
    # 返回状态字典和行文本
    states = {i: (0 if ch == 'B' else 1) for i, ch in enumerate(flat) if ch in 'BW'}
    return states, rows

# 获取以落子点为中心的邻域区域（7x7）
def generate_meaning_moves(stones: set, avail: set):
    out = set()
    for p in stones:
        r0, c0 = divmod(p, BOARD)
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                r, c = r0 + dr, c0 + dc
                idx = r * BOARD + c
                if 0 <= r < BOARD and 0 <= c < BOARD and idx not in stones and idx in avail:
                    out.add(idx)
    return list(out)

# ---------- 将单个棋盘字符串转为特征张量和标签 ----------
def board_to_feature(key: str, label: int):
    states, rows = deserialize_board(key)
    current_player = len(states) % 2

    epm = gomoku_engine.Board()
    epm.set_board(rows)

    # 可落子点、valuable点、wise点、meaning点
    avail = epm.available() if current_player == 0 else [i for i in range(BOARD * BOARD) if i not in states]
    _, valuable = epm.valuable()
    wise = epm.wise()
    meaning = generate_meaning_moves(set(states), set(avail))

    # 初始化7通道特征
    feat = np.zeros((PLANES, BOARD, BOARD), np.float32)
    moves, players = zip(*states.items()) if states else ([], [])
    moves, players = np.array(moves), np.array(players)
    blk, wht = moves[players == 0], moves[players == 1]

    # 0通道: 当前黑棋位置，1通道: 当前白棋位置，2通道: 当前轮到谁
    if current_player == 0:
        feat[0][blk // BOARD, blk % BOARD] = 1
        feat[1][wht // BOARD, wht % BOARD] = 1
        feat[2][:] = 1  # 黑走
    else:
        feat[0][wht // BOARD, wht % BOARD] = 1
        feat[1][blk // BOARD, blk % BOARD] = 1
        feat[2][:] = 0  # 白走

    # 3通道: valuable点，4通道: wise点，5通道: meaning点，6通道: 所有可落子点
    for i, arr in enumerate([valuable, wise, meaning, avail]):
        idx = np.array(arr)
        feat[3+i][idx // BOARD, idx % BOARD] = 1

    return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ---------- 自定义 IterableDataset，用于轮流采样 ----------
class UTSSOnlineDataset(IterableDataset):
    def __init__(self, hash_paths: dict):
        self.tables = {label: list(load_hash_table(path).keys()) for label, path in hash_paths.items()}
        self.indices = {label: 0 for label in hash_paths}

    def __iter__(self):
        while True:
            # 按照 0→2→1→3 顺序循环取样
            for label in [0, 2, 1, 3]:
                keys = self.tables[label]
                i = self.indices[label]
                if i >= len(keys): continue
                self.indices[label] += 1
                yield board_to_feature(keys[i], label)

# ---------- 主训练流程 ----------
def train():
    dataset = UTSSOnlineDataset(hash_paths)                       # 动态生成样本的 Dataset
    dataloader = DataLoader(dataset, batch_size=BATCH, num_workers=0)

    # 加载三头网络（输入通道=7）
    net = Net(in_channels=7, num_channels=128, num_res_blocks=7, board_size=15).to(DEVICE)

    # 仅训练 utss 分支的参数，冻结其他参数
    for n, p in net.named_parameters():
        p.requires_grad = n.startswith("utss")

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3)

    step = 0
    net.train()
    for epoch in range(1, EPOCH + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", total=MAX_STEPS, mininterval=1.0)
        for xb, yb in pbar:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            _, _, logits = net(xb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            preds = logits.argmax(1)
            acc = (preds == yb).float().mean().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")

            step += 1
            if step >= MAX_STEPS:
                torch.save(net.state_dict(), "utss_pretrain_online.pt")
                print("✓ 模型已保存为 utss_pretrain_online.pt")
                return

# ---------- 脚本入口 ----------
if __name__ == "__main__":
    train()
