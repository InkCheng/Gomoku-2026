# === pretrain_utss_head.py ==========================================
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
from src.models.policy_value_utss_net import Net
from tqdm import tqdm

# ---------- 可调超参 ----------
BATCH      = 512
LR         = 1e-3
EPOCH_MAX  = 10
PATIENCE   = 2
CHUNK_DIR  = Path("utss_fast_chunks")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 1.  加载数据集 ----------
class ChunkDataset(Dataset):
    def __init__(self, files):
        self.files = files
        self.idx2file, self.offsets = [], []
        for f in files:
            y = torch.load(f, map_location="cpu")[1]
            self.idx2file.extend([f]*len(y))
            self.offsets.extend(torch.arange(len(y)).tolist())
        self.idx2file, self.offsets = map(np.asarray, (self.idx2file, self.offsets))

    def __len__(self): return len(self.idx2file)

    def __getitem__(self, idx):
        f, off = self.idx2file[idx], self.offsets[idx]
        X, y = torch.load(f, map_location="cpu")
        X = X[off]  # (C=3, 15, 15) or (7, 15, 15)
        ### 只有 3 通道，就在 channel 维度 pad 4 个 0 ----
        if X.shape[0] == 3:  # 只有 3 通道 → 拼 4 个全 0 通道
            pad = torch.zeros(4, *X.shape[1:], dtype=X.dtype)
            X = torch.cat([X, pad], dim=0)  # (7, H, W)
        ### -----------------------------------------------------------------
        return X, y[off]

files   = sorted(glob(str(CHUNK_DIR / "chunk_*.pt")))
assert files, f"找不到 {CHUNK_DIR}/chunk_*.pt"                # 防呆
full_ds = ChunkDataset(files)

# ---------- 2.  分层划分 train / val / test ----------
labels  = torch.cat([torch.load(f, map_location="cpu")[1] for f in files])
idx_all = np.arange(len(labels))

train_idx, rest_idx = train_test_split(idx_all, test_size=0.30,
                                       stratify=labels, random_state=42)
val_idx, test_idx = train_test_split(rest_idx, test_size=0.50,
                                     stratify=labels[rest_idx], random_state=42)

dl_train = DataLoader(Subset(full_ds, train_idx), BATCH, shuffle=True,
                      num_workers=0, pin_memory=True)
dl_val   = DataLoader(Subset(full_ds, val_idx),   BATCH, shuffle=False,
                      num_workers=0, pin_memory=True)
dl_test  = DataLoader(Subset(full_ds, test_idx),  BATCH, shuffle=False,
                      num_workers=0, pin_memory=True)

print(f"# samples  train {len(train_idx)}  val {len(val_idx)}  test {len(test_idx)}")

# ---------- 3.  建网络并冻结非 TSS 参数 ----------
net = Net(in_channels=7, num_channels=128, board_size=15).to(DEVICE)
for n, p in net.named_parameters():
    p.requires_grad = n.startswith("utss")

opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                        lr=LR, weight_decay=1e-3)

# ---------- 4.  训练 + 早停 ----------
best_f1 = patience = 0
for epoch in range(1, EPOCH_MAX + 1):
    net.train()
    for xb, yb in tqdm(dl_train, desc=f"epoch {epoch}", leave=False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        _, _, logits = net(xb)                    # 删掉 use_tss=True
        loss = F.cross_entropy(logits, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

    # ---- 验证 ----
    net.eval()
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in dl_val:
            xb = xb.to(DEVICE)
            logits = net(xb)[2]
            ys.extend(yb.tolist())
            preds.extend(logits.argmax(1).cpu().tolist())

    f1 = f1_score(ys, preds, average="macro")
    print(f"epoch {epoch:2d}  val F1={f1:.4f}")

    if f1 > best_f1:
        best_f1, patience = f1, 0
        torch.save(net.state_dict(), "utss_pretrain.pt")
        print("  ✓ saved utss_pretrain.pt")
    else:
        patience += 1
    if patience >= PATIENCE or best_f1 >= 0.92:
        print(">>> early stop.")
        break

# ---------- 5.  测试评估 ----------
net.load_state_dict(torch.load("utss_pretrain.pt"), strict=False)
net.eval()
ys, preds = [], []
with torch.no_grad():
    for xb, yb in dl_test:
        xb = xb.to(DEVICE)
        logits = net(xb)[2]
        ys.extend(yb.tolist())
        preds.extend(logits.argmax(1).cpu().tolist())

test_f1 = f1_score(ys, preds, average="macro")
print(f"\nFinal Test macro-F1 : {test_f1:.4f}")
# ===================================================================
