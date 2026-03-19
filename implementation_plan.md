# 重构 UI.py → UI2.py（不依赖 pygame / gomoku_engine，加入多线程）

## 背景

原 [UI.py](file:///f:/code/Gomoku_Competition_V1.0.3/Gomoku_Competition_V1.0.3/UI.py) 使用 **pygame** 渲染棋盘，使用 **gomoku_engine**（C++ 编译的 [.pyd](file:///f:/code/Gomoku_Competition_V1.0.3/Gomoku_Competition_V1.0.3/gomoku_engine.pyd)）进行禁手判断与 TSS 搜索。
需要新建 `UI2.py`，实现相同的人机对弈功能，但改用 **tkinter + 纯 Python 棋盘逻辑 + 多线程**。

## Proposed Changes

### [NEW] [UI2.py](file:///f:/code/Gomoku_Competition_V1.0.3/Gomoku_Competition_V1.0.3/UI2.py)

一个完整的、自包含的文件，包含以下主要组件：

---

#### 1. `SimpleBoard` — 纯 Python 棋盘逻辑

复刻 `game.Board` 的核心接口，但 **不依赖 `gomoku_engine`** ：

| 方法 | 说明 |
|---|---|
| [init_board(start_player)](file:///f:/code/Gomoku_Competition_V1.0.3/Gomoku_Competition_V1.0.3/src/game.py#202-214) | 初始化棋盘 |
| [do_move(move)](file:///f:/code/Gomoku_Competition_V1.0.3/Gomoku_Competition_V1.0.3/src/game.py#431-447) | 下子并切换玩家 |
| [undo_move(x, y)](file:///f:/code/Gomoku_Competition_V1.0.3/Gomoku_Competition_V1.0.3/src/game.py#448-465) | 悔棋 |
| [move_to_location(move)](file:///f:/code/Gomoku_Competition_V1.0.3/Gomoku_Competition_V1.0.3/src/game.py#316-321) / [location_to_move(loc)](file:///f:/code/Gomoku_Competition_V1.0.3/Gomoku_Competition_V1.0.3/src/game.py#322-334) | 坐标转换 |
| [game_end()](file:///f:/code/Gomoku_Competition_V1.0.3/Gomoku_Competition_V1.0.3/src/game.py#231-241) / [has_a_winner()](file:///f:/code/Gomoku_Competition_V1.0.3/Gomoku_Competition_V1.0.3/src/game.py#466-476) | 胜负判定 |
| [connect_5(move, color)](file:///f:/code/Gomoku_Competition_V1.0.3/Gomoku_Competition_V1.0.3/src/game.py#267-291) | 检查五连 |
| [model_current_state()](file:///f:/code/Gomoku_Competition_V1.0.3/Gomoku_Competition_V1.0.3/src/game.py#416-429) | 返回 3×15×15 状态给模型 |
| `availables` 属性 | 返回所有空位（简化版，不做禁手检查） |

> [!NOTE]
> 去掉 `gomoku_engine` 后不做黑棋禁手检查（长连/四四/三三），所有空位均可落子。这是与原版最大的区别。

---

#### 2. `SimpleAI` — 只用策略网络的 AI

- 用 `torch.jit.load()` 加载 [best_model/current_policy_step_best.model](file:///f:/code/Gomoku_Competition_V1.0.3/Gomoku_Competition_V1.0.3/best_model/current_policy_step_best.model)
- 调用 [model_current_state()](file:///f:/code/Gomoku_Competition_V1.0.3/Gomoku_Competition_V1.0.3/src/game.py#416-429) 构造输入张量
- 从模型输出的策略概率中选最高合法位置作为落子
- **不使用 MCTS**（MCTS 深度依赖 `gomoku_engine` 的 TSS/valuable/wise 方法）

> [!IMPORTANT]
> 棋力会弱于原版（原版有 800 次 MCTS 模拟 + TSS 搜索），但可以正常下棋。

---

#### 3. `GomokuApp` — tkinter 界面

| 组件 | 说明 |
|---|---|
| `tk.Canvas` | 绘制 15×15 棋盘网格和棋子（圆形） |
| 状态栏 | 显示当前回合、AI 思考中提示 |
| 按钮区 | 悔棋、新游戏、先手选择 |
| 棋谱保存 | 落子记录保存至 `棋谱/` 目录 |

棋盘使用 **Canvas 绘制**：木色背景 + 网格线 + 星位点 + 黑/白圆形棋子 + 最后落子红色标记。

---

#### 4. 多线程

- AI 落子在 `threading.Thread` 中运行，避免阻塞主线程
- 使用 `root.after()` 轮询线程结果，更新 UI
- 思考中禁止用户点击棋盘

---

## Verification Plan

### Manual Verification

由于该项目依赖 GPU 模型文件和 tkinter 窗口，最适合手动测试：

1. 在项目目录运行 `python UI2.py`
2. 确认弹出 tkinter 窗口，显示 15×15 棋盘
3. 选择先手方（AI 先手 / 人类先手）
4. 点击棋盘落子，确认黑子出现在正确位置
5. 确认 AI 在后台线程中思考，状态栏显示"AI思考中..."
6. AI 落子后白子出现，状态栏切换回人类回合
7. 测试悔棋按钮（应撤销两步：AI 一步 + 人类一步）
8. 下到五连后确认弹出胜负提示
