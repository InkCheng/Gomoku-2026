# Gomoku_Competition

## 项目结构

```
Gomoku_Competition_V1.0.3/
├── UI.py                          # 用户界面 (必须保留在根目录)
├── train.py                       # 训练脚本 (必须保留在根目录)
├── model/                         # 训练模型输出目录 (严禁改动)
├── best_model/                    # 最佳模型存储
│   ├── tss_classifier_800.pth      # TSS分类器权重
│   ├── current_policy_step_best.model  # 最佳策略模型
│   └── ...
├── data/                          # 资源文件
│   ├── bg_old.png                 # 背景图
│   ├── chess_black.png            # 黑棋图片
│   └── ...
├── font/                          # 字体文件
├── 棋谱/                         # 棋谱存储目录
├── src/                           # 核心源代码
│   ├── game.py                    # 游戏核心逻辑
│   │   ├── Board                  # 棋盘类
│   │   ├── Game                   # 游戏控制类
│   │   └── HashTableManager       # 哈希表管理器
│   ├── players.py                 # 玩家类
│   │   ├── AIplayer              # AI玩家
│   │   ├── Human                 # 人类玩家
│   │   └── AIPlayer_MCTS         # MCTS AI玩家
│   ├── mcts/                      # MCTS搜索算法
│   │   ├── mcts_alphazero1.py     # MCTS算法v1 (基础版本)
│   │   ├── mcts_alphazero2.py     # MCTS算法v2 (优化版本)
│   │   └── MCTS_alphazero_test.py # MCTS测试工具
│   └── models/                    # 深度学习模型
│       ├── policy_value_net.py     # 策略价值网络 (双头)
│       ├── policy_value_utss_net.py # UTSS策略价值网络 (三头)
│       ├── tss_classifier.py      # TSS分类器 (新版本)
│       └── tss_classifier_old.py  # TSS分类器 (旧版本)
├── tests/                         # 测试和评估脚本
│   ├── pk.py                      # 对战测试脚本
│   └── Elo_rate.py                # ELO评级脚本
├── utils/                         # 工具和辅助函数
│   ├── preDataset.py              # 数据预处理工具
│   └── reverse.py                 # 数据反转工具
└── config/                        # 配置文件
    ├── tss_pretrain.py            # TSS预训练配置
    └── utss_weight_scheduler.py   # UTSS权重调度配置
```

#### 分层

- **游戏逻辑层**: `src/game.py`, `src/players.py`
- **算法层**: `src/mcts/`, `src/models/`
- **测试层**: `tests/`
- **工具层**: `utils/`
- **配置层**: `config/`

#### 功能聚合

- MCTS相关 → `src/mcts/`
- 神经网络模型 → `src/models/`
