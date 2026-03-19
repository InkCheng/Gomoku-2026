# 项目结构说明

## 整理后的项目结构

```
Gomoku_Competition_V1.0.3/
├── UI.py                          # 用户界面 (必须保留在根目录)
├── train.py                       # 训练脚本 (必须保留在根目录)
├── model/                         # 模型文件夹 (严禁改动)
├── best_model/                    # 最佳模型文件
├── data/                          # 数据文件 (棋子图片、背景等)
├── font/                          # 字体文件
├── 棋谱/                         # 棋谱目录
├── src/                           # 源代码目录
│   ├── __init__.py
│   ├── game.py                    # 游戏核心逻辑 (Board, Game, HashTableManager)
│   ├── players.py                 # 玩家相关 (AIplayer, Human, AIPlayer_MCTS)
│   ├── mcts/                      # MCTS相关模块
│   │   ├── __init__.py
│   │   ├── mcts_alphazero1.py     # MCTS算法1 (旧版本)
│   │   ├── mcts_alphazero2.py     # MCTS算法2 (新版本)
│   │   └── MCTS_alphazero_test.py # MCTS测试
│   └── models/                    # AI模型相关模块
│       ├── __init__.py
│       ├── policy_value_net.py     # 策略价值网络
│       ├── policy_value_utss_net.py # UTSS策略价值网络
│       ├── tss_classifier.py      # TSS分类器
│       └── tss_classifier_old.py  # TSS分类器旧版本
├── tests/                         # 测试和评估文件
│   ├── __init__.py
│   ├── pk.py                      # 对战脚本
│   └── Elo_rate.py                # ELO评级
├── utils/                         # 工具和辅助文件
│   ├── __init__.py
│   ├── preDataset.py              # 数据集预处理
│   └── reverse.py                 # 反转功能
└── config/                        # 配置文件
    ├── __init__.py
    ├── tss_pretrain.py            # TSS预训练配置
    └── utss_weight_scheduler.py   # UTSS权重调度
```

## 文件移动和导入路径更新

### 核心游戏逻辑 → src/
- `game.py` → `src/game.py`
- `players.py` → `src/players.py`

### AI/ML相关 → src/models/
- `policy_value_net.py` → `src/models/policy_value_net.py`
- `policy_value_utss_net.py` → `src/models/policy_value_utss_net.py`
- `tss_classifier.py` → `src/models/tss_classifier.py`
- `tss_classifier_old.py` → `src/models/tss_classifier_old.py`

### MCTS相关 → src/mcts/
- `mcts_alphazero1.py` → `src/mcts/mcts_alphazero1.py`
- `mcts_alphazero2.py` → `src/mcts/mcts_alphazero2.py`
- `MCTS_alphazero_test.py` → `src/mcts/MCTS_alphazero_test.py`

### 测试和评估 → tests/
- `pk.py` → `tests/pk.py`
- `Elo_rate.py` → `tests/Elo_rate.py`

### 工具和辅助 → utils/
- `preDataset.py` → `utils/preDataset.py`
- `reverse.py` → `utils/reverse.py`

### 配置 → config/
- `tss_pretrain.py` → `config/tss_pretrain.py`
- `utss_weight_scheduler.py` → `config/utss_weight_scheduler.py`

## 导入路径更新示例

### 更新前
```python
from game import Board, Game, HashTableManager
from players import AIplayer
from mcts_alphazero2 import MCTSPlayer as MCTSPlayer2
from policy_value_utss_net import PolicyValueUTSSNet
from tss_classifier_old import TSSClassifier
```

### 更新后
```python
from src.game import Board, Game, HashTableManager
from src.players import AIplayer
from src.mcts.mcts_alphazero2 import MCTSPlayer as MCTSPlayer2
from src.models.policy_value_utss_net import PolicyValueUTSSNet
from src.models.tss_classifier_old import TSSClassifier
```

## 运行项目

1. **训练模型**:
   ```bash
   python train.py
   ```

2. **运行UI界面**:
   ```bash
   python UI.py
   ```

3. **测试对战**:
   ```bash
   python tests/pk.py
   ```

4. **ELO评级**:
   ```bash
   python tests/Elo_rate.py
   ```

## 注意事项

1. **UI.py** 和 **train.py** 必须保留在根目录下
2. **model** 文件夹的位置和内容严禁改动
3. 所有文件的导入路径已更新，无需手动修改
4. 如需添加新的模块，请按照现有结构进行组织

## 整理优势

1. **清晰的模块划分**: 按功能将代码分为不同目录
2. **易于维护**: 相关代码集中在一起，便于查找和修改
3. **便于扩展**: 新增功能时可以根据类型放入对应目录
4. **减少混乱**: 避免根目录文件过多，提高可读性