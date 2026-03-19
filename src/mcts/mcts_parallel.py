"""
mcts_parallel.py — 多线程树并行 MCTS 搜索

参照 Gamma Connect6 的 Tree Parallelization 设计:
- 多个 worker 线程共用一棵搜索树
- Virtual Loss 鼓励探索不同分支
- 节点锁保护 expand / update
- 通过 NNBatchServer 异步批量推理
"""

import numpy as np
import copy
import time
import threading
import torch
from concurrent.futures import Future


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


# ─── 线程安全的树节点 ─────────────────────────────────────────

class TreeNodeParallel:
    """带锁的 MCTS 树节点, 支持多线程并发访问。"""

    __slots__ = (
        '_parent', '_children', '_n_visits', '_Q', '_u', '_P',
        '_virtual_loss', '_lock', '_tss_checked'
    )

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0.0
        self._u = 0.0
        self._P = prior_p
        self._virtual_loss = 0
        self._lock = threading.Lock()
        self._tss_checked = False

    # ── select (读操作, 不需要独占锁) ──
    def select(self, c_puct):
        """选择 UCB 值最大的子节点。"""
        with self._lock:
            children_snapshot = list(self._children.items())
        if not children_snapshot:
            return None, None
        return max(children_snapshot,
                   key=lambda act_node: act_node[1].get_value(c_puct))

    # ── expand (写操作, 需要独占锁) ──
    def expand(self, action_priors, board):
        """扩展叶节点。"""
        with self._lock:
            board.externalProgramManager.set_board(board.serialize_board())
            value_flag, valuable_moves = board.externalProgramManager.valuable()

            if value_flag and valuable_moves:
                for action, prob in action_priors:
                    if action in valuable_moves and action not in self._children:
                        self._children[action] = TreeNodeParallel(self, prob)

            if not value_flag or not valuable_moves:
                wise_move = _get_wisemove(board)
                if wise_move:
                    for action, prob in action_priors:
                        if action in wise_move and action not in self._children:
                            self._children[action] = TreeNodeParallel(self, prob)
                else:
                    for action, prob in action_priors:
                        if action not in self._children:
                            self._children[action] = TreeNodeParallel(self, prob)

            if not self._children:
                for action, prob in action_priors:
                    if action not in self._children:
                        self._children[action] = TreeNodeParallel(self, prob)

    # ── update (短锁保护统计数据) ──
    def update(self, leaf_value):
        with self._lock:
            self._n_visits += 1
            self._Q += (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    # ── Virtual Loss ──
    def apply_virtual_loss(self, vl=3):
        """选中时施加虚拟损失, 避免多线程扎堆同一路径。"""
        with self._lock:
            self._virtual_loss += vl
            self._n_visits += vl
            # 降低 Q 值: 假设虚拟损失是输 (对当前方不利)
            self._Q = (self._Q * (self._n_visits - vl) - vl) / self._n_visits

    def revert_virtual_loss(self, vl=3):
        """回退虚拟损失。"""
        with self._lock:
            self._n_visits -= vl
            self._virtual_loss -= vl
            if self._n_visits > 0:
                self._Q = (self._Q * (self._n_visits + vl) + vl) / self._n_visits
            else:
                self._Q = 0.0

    def get_value(self, c_puct):
        parent_visits = self._parent._n_visits if self._parent else 1
        self._u = c_puct * self._P * np.sqrt(parent_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def is_leaf(self):
        return len(self._children) == 0

    def is_root(self):
        return self._parent is None


# ─── 并行 MCTS ─────────────────────────────────────────────────

class MCTSParallel:
    """
    多线程树并行 MCTS。

    多个 worker 并发执行 playout, 共享同一棵树。
    通过 NNBatchServer 异步批量推理。
    """

    def __init__(self, model=None, tss_classifier=None,
                 nn_batch_server=None,
                 c_puct=5, n_playout=10000,
                 num_workers=4, virtual_loss=3):
        self.model = model
        self.tss_classifier = tss_classifier
        self.nn_batch_server = nn_batch_server
        self._root = TreeNodeParallel(None, 1.0)
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.num_workers = num_workers
        self.virtual_loss = virtual_loss

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _evaluate_leaf(self, state):
        """
        通过 NNBatchServer 异步推理, 如果 server 不可用则同步推理。
        """
        state_np = state.model_current_state()

        if self.nn_batch_server is not None:
            # 异步批量推理
            fut = self.nn_batch_server.predict_commit(state_np)
            act_probs_flat, value = fut.result()  # 阻塞等待结果
        else:
            # 同步推理 (fallback)
            state_tensor = torch.from_numpy(state_np).to(
                self.device).unsqueeze(0).float()
            with torch.no_grad():
                log_act_probs, value_t = self.model(state_tensor)
            act_probs_flat = np.exp(
                log_act_probs.detach().cpu().numpy().flatten()
            )
            value = value_t.item()

        # 将不可用位置概率置零
        available = state.availables
        for i in range(len(act_probs_flat)):
            if i not in available:
                act_probs_flat[i] = 0.0

        total = np.sum(act_probs_flat)
        if total > 0:
            act_probs_flat /= total

        return list(enumerate(act_probs_flat)), value

    def _playout(self, state):
        """单次 playout: select → expand → evaluate → backprop"""
        node = self._root
        path = []  # 记录路径用于回退 virtual loss

        # ── Select ──
        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            if node is None:
                break
            node.apply_virtual_loss(self.virtual_loss)
            path.append(node)
            state.do_move(action)

        # ── TSS / 哈希表检查 ──
        flag = 0
        current_state_str = str(state.serialize_board())
        if current_state_str in state.hash_table_manager.hash_table:
            flag = 1

        if flag == 1:
            end = True
            winner = state.get_current_player()
        else:
            end, winner = state.game_end()

        # ── Evaluate / Expand ──
        if not end:
            action_probs, leaf_value = self._evaluate_leaf(state)
            node.expand(action_probs, state)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # ── Backpropagate ──
        node.update_recursive(-leaf_value)

        # ── 回退 Virtual Loss ──
        for n in path:
            n.revert_virtual_loss(self.virtual_loss)

    def _worker_loop(self, state_template, stop_event, playout_counter):
        """
        Worker 线程主循环。

        每次 deepcopy 棋盘状态后执行一次 playout。
        """
        while not stop_event.is_set():
            # 原子递增 playout 计数
            with playout_counter['lock']:
                if playout_counter['count'] >= self._n_playout:
                    break
                playout_counter['count'] += 1

            state_copy = copy.deepcopy(state_template)
            try:
                self._playout(state_copy)
                # 让出 GIL，防止 Python 多线程死锁阻塞 PyQt 事件循环
                time.sleep(0.001) 
            except Exception as e:
                print(f"[MCTS-Worker] playout 异常: {e}")

    def get_move_probs(self, state, temp=1e-3, time_limit=15):
        """
        多线程并行搜索, 返回 (acts, act_probs)。
        """
        print(f"[MCTS-Parallel] 启动 {self.num_workers} 个 worker, "
              f"目标 {self._n_playout} 次 playout, 时限 {time_limit}s")

        stop_event = threading.Event()
        playout_counter = {'count': 0, 'lock': threading.Lock()}

        start_time = time.time()

        # 启动 worker 线程
        workers = []
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                args=(state, stop_event, playout_counter),
                daemon=True,
                name=f"MCTS-Worker-{i}"
            )
            t.start()
            workers.append(t)

        # 主线程监控时间
        while True:
            elapsed = time.time() - start_time
            if elapsed >= time_limit:
                stop_event.set()
                print(f"[MCTS-Parallel] 超时停止, 已用 {elapsed:.1f}s")
                break
            with playout_counter['lock']:
                if playout_counter['count'] >= self._n_playout:
                    break
            time.sleep(0.05)

        stop_event.set()
        for t in workers:
            t.join(timeout=2.0)

        with playout_counter['lock']:
            total_playouts = playout_counter['count']
        elapsed = time.time() - start_time
        print(f"[MCTS-Parallel] 完成 {total_playouts} 次 playout, "
              f"耗时 {elapsed:.2f}s, "
              f"速率 {total_playouts / max(elapsed, 0.001):.0f} playouts/s")

        # ── 计算移动概率 ──
        with self._root._lock:
            act_visits = [(act, node._n_visits)
                          for act, node in self._root._children.items()]

        if not act_visits:
            print("[MCTS-Parallel] 警告: 没有可用动作, 重试")
            return self.get_move_probs(state, temp, time_limit)

        acts, visits = zip(*act_visits)
        valid_moves = state.availables
        act_probs = np.zeros(len(acts))
        valid_indices = [i for i in range(len(acts)) if acts[i] in valid_moves]

        if valid_indices:
            valid_visits = [visits[i] for i in valid_indices]
            valid_probs = softmax(
                1.0 / temp * np.log(np.array(valid_visits) + 1e-10)
            )
            act_probs[valid_indices] = valid_probs

        return acts, act_probs

    def get_win_rate(self):
        return self._root._Q

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNodeParallel(None, 1.0)

    def __str__(self):
        return "MCTS-Parallel"


# ─── 并行 MCTSPlayer ──────────────────────────────────────────

class MCTSPlayerParallel:
    """基于并行 MCTS 的 AI 玩家"""

    def __init__(self, c_puct=5, n_playout=400, is_selfplay=0,
                 model=None, tss_classifier=None,
                 nn_batch_server=None,
                 num_workers=4, virtual_loss=3):
        self.model = model
        self.tss_classifier = tss_classifier
        self.mcts = MCTSParallel(
            model=model,
            tss_classifier=tss_classifier,
            nn_batch_server=nn_batch_server,
            c_puct=c_puct,
            n_playout=n_playout,
            num_workers=num_workers,
            virtual_loss=virtual_loss,
        )
        self._is_selfplay = is_selfplay
        self.search_time = 15  # 默认搜索时间

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        move_probs = np.zeros(board.width * board.height)

        tss_flag = 0
        flag = 0

        if len(sensible_moves) > 0:
            # ── TSS 辅助搜索 ──
            if len(sensible_moves) < 223:
                current_state = str(board.serialize_board())
                if current_state in board.hash_table_manager.hash_table:
                    print("[MCTS-Parallel] 在 hash 表中找到最优点")
                    move = board.hash_table_manager.hash_table[current_state]
                    move_probs[move] = 1
                    tss_flag = 1
                else:
                    board.externalProgramManager.set_board(
                        board.serialize_board()
                    )
                    result = board.externalProgramManager.tss(10)
                    if result is not None and result[0] == 1:
                        print("[MCTS-Parallel] TSS 发现必胜点: "
                              + str(board.move_to_location(result[1])))
                        move = result[1]
                        move_probs[move] = 1
                        tss_flag = 1
                        board.hash_table_manager.add(current_state, move)
                    elif result is not None and result[0] == -2:
                        board.hash_table_manager.add_limited(
                            current_state, -2
                        )
                        board.hash_table_manager.add_check(
                            current_state, -1
                        )
                    elif result is not None and result[0] == -1:
                        board.hash_table_manager.add_check(
                            current_state, -1
                        )

            # ── MCTS 搜索 ──
            if tss_flag == 0:
                board.externalProgramManager.set_board(
                    board.serialize_board()
                )
                value_flag, valuable_moves = (
                    board.externalProgramManager.valuable()
                )

                if len(valuable_moves) == 1:
                    move = valuable_moves[0]
                    move_probs[move] = 1
                    flag = 1
                else:
                    time_limit = getattr(self, "search_time", 15)
                    acts, probs = self.mcts.get_move_probs(
                        board, temp, time_limit=time_limit
                    )
                    move_probs = np.zeros(board.width * board.height)
                    move_probs[list(acts)] = probs

                    wise_move = _get_wisemove(board)
                    wise_probs = np.zeros(len(acts))
                    mask = np.array(
                        [act in wise_move for act in acts]
                    )
                    wise_probs[mask] = probs[mask]
                    wise_sum = np.sum(wise_probs)
                    if wise_sum > 0:
                        wise_probs /= wise_sum
                    else:
                        wise_probs[:] = 1.0 / len(wise_probs)

                    acts = [act for act in acts if act in wise_move]
                    probs = wise_probs[mask]
                    move_probs = np.zeros(board.width * board.height)
                    move_probs[acts] = probs

            # ── 选择落子 ──
            if self._is_selfplay:
                if tss_flag == 0 and flag == 0:
                    p = 0.75 * probs + 0.25 * np.random.dirichlet(
                        0.3 * np.ones(len(probs))
                    )
                    move = np.random.choice(acts, p=p)
                self.mcts.update_with_move(move)
            else:
                if tss_flag == 0 and flag == 0:
                    probs_sum = np.sum(probs)
                    if probs_sum > 0:
                        probs = probs / probs_sum
                    else:
                        probs[:] = 1.0 / len(probs)
                    move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("[MCTS-Parallel] 棋盘已满")

    def __str__(self):
        return "MCTS-Parallel-Player"


# ─── 工具函数 ──────────────────────────────────────────────────

def _get_wisemove(board):
    """获取棋盘上已有棋子周围 3 格内的空位。"""
    res = []
    for pos in board.states.keys():
        for i in range(-3, 4):
            for j in range(-3, 4):
                new_pos = pos + i * board.width + j
                if (0 <= new_pos < board.width * board.height
                        and new_pos not in res
                        and new_pos not in board.states
                        and new_pos in board.availables):
                    res.append(new_pos)
    return res
