import numpy as np
import copy
import time
import torch

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def get_equi_data_tss(state, tss_flag):
        """通过旋转和翻转来增加TSS分类器的训练数据"""
        extend_data = []
        state = np.array(state)
        for i in [1, 2, 3, 4]:
            # 逆时针旋转
            equi_state = np.array([np.rot90(s, i) for s in state])
            extend_data.append((equi_state,tss_flag))
            # 水平翻转
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            extend_data.append((equi_state, tss_flag))
        return extend_data

class TreeNode(object):
    """MCTS树中的节点。

    每个节点跟踪其自身的值Q,先验概率P及其访问次数调整的先前得分u。
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # 从动作到TreeNode的映射
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors, board):
        """通过创建新子项来展开树。
        action_priors:一系列动作元组及其先验概率根据策略函数.
        board: 当前的棋盘状态
        """
        board.externalProgramManager.set_board(board.serialize_board())
        value_flag,valuable_moves = board.externalProgramManager.valuable()  # 威胁走法（成五点/冲四点/冲三点/破着点）
        if value_flag:
            if valuable_moves:
                for action, prob in action_priors:
                    if action in valuable_moves and action not in self._children:
                        self._children[action] = TreeNode(self, prob)  # 加入子节点

        if not value_flag or not valuable_moves:  # 没有威胁走法
            wise_move = get_meaningmove(board)  # 走法过滤
            if wise_move:
                for action, prob in action_priors:
                    if action in wise_move and action not in self._children:
                        self._children[action] = TreeNode(self, prob)  # 周围三格的棋子加入子节点
            else:
                # 如果没有wise_move,则按原来的方式展开所有动作
                for action, prob in action_priors:
                    if action not in self._children:
                        self._children[action] = TreeNode(self, prob)  # 周围三格不能落子，则展开其他动作

        if self._children=={}:
            #警告
            print("警告：self._children所有动作的概率都是0")
             # 如果没有wise_move,则按原来的方式展开所有动作
            for action, prob in action_priors:
                if action not in self._children:
                    self._children[action] = TreeNode(self, prob)  # 防止children为空

    def select(self, c_puct,board=None):
        """在子节点中选择能够提供最大行动价值Q的行动加上奖金u(P)。
            return:(action,next_node)的元组
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """从叶节点评估中更新节点值
        leaf_value: 这个子树的评估值来自从当前玩家的视角
        """
        # 统计访问次数
        self._n_visits += 1
        # 更新Q值,取对于所有访问次数的平均数
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """就像调用update()一样,但是对所有祖先进行递归应用。
        """
        # 如果它存在父节点（即不是根节点）,则应首先更新此节点的父节点。
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """计算并返回此节点的值。它是叶评估Q和此节点的先验的组合
            调整了访问次数,u。
            c_puct:控制相对影响的(0,inf)中的数字,该节点得分的值Q和先验概率P.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """检查叶节点(即没有扩展的节点)。"""
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS(object):
    """对蒙特卡罗树搜索的一个简单实现"""

    def __init__(self, policy_value_tss_fn, c_puct=5, n_playout=10000, selfplay=False):
        """
        policy_value_fn:一个接收板状态和输出的函数(动作,概率)元组列表以及[-1,1]中的分数
             (即来自当前的最终比赛得分的预期值玩家的观点)对于当前的玩家。
    c_puct:(0,inf)中的数字,用于控制探索的速度收敛于最大值政策。 更高的价值意味着
             依靠先前的更多。
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_tss_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.is_selfplay = selfplay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.total_nodes = 0
        self.total_branches = 0

    def _playout(self, state):
        # 执行一次树搜索过程
        node = self._root
        depth = 0
        # 选择
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct, state)
            state.do_move(action)
            depth += 1
        # 扩展和评估
        current_state = str(state.serialize_board())
        tss_flag = -1
            # 查哈希表
        if current_state in state.hash_table_manager.hash_table.keys():
            tss_flag = 0  # 必胜
        elif current_state in state.hash_table_manager.loss_table.keys():
            tss_flag = 2  # 必败
        elif current_state in state.hash_table_manager.check_table.keys():
            tss_flag = 1  # 平局
        elif current_state in state.hash_table_manager.limited_time_hash.keys():
            tss_flag = 3  # 超时

            # 未命中，调用UTSS分类头
        use_utss = (tss_flag == -1 and self.time_flag == 0 and state.turn >= 10)

        with torch.no_grad():
            action_probs, leaf_value, utss_logits = self._policy(state, use_tss=use_utss)

        if use_utss and utss_logits is not None:
            utss_probs = torch.softmax(utss_logits, dim=1)
            # print(f"utss_probs: {utss_probs}")  # 打印所有4类概率
            winning_prob = utss_probs[0, 0].item()
            losing_prob = utss_probs[0, 2].item()
            if (winning_prob > 0.6 or losing_prob > 0.6):
                print(f"UTSS头检测到叶子节点可能是必胜必败局面，概率为{winning_prob:.2f}，{losing_prob:.2f}")
                print("深度为" + str(depth) + "时,进行tss搜索")
                print(self.tss_time)
                self.tss_time += 1

                state.externalProgramManager.set_board(state.serialize_board())
                result = state.externalProgramManager.tss(5)

                if result is not None:
                    if result[0] == 1:
                        tss_flag = 0  # 必胜
                        print("叶子节点出现必胜点:" + str(state.move_to_location(result[1])))
                        state.hash_table_manager.add(current_state, result[1])
                    elif result[0] == -1:
                        tss_flag = 2  # 必败
                        state.hash_table_manager.add_loss(current_state, -1)
                    elif result[0] == 0:
                        tss_flag = 1  # 平局
                        state.hash_table_manager.add_check(current_state, 0)
                    elif result[0] == -2:
                        tss_flag = 3  # 未知
                        state.hash_table_manager.add_limited(current_state, -2)

            # 终局状态，直接设置终局信息；否则展开节点
        if tss_flag == 0:  # 必胜到达终局剪枝
            end = 1
            winner = state.get_current_player()
        elif tss_flag == 2:  # 必败到达终局剪枝
            end = 1
            winner = 1 - state.get_current_player()
        else:
            end, winner = state.game_end()

        if not end:
            node.expand(action_probs, state)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0
        # 回溯
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3,time_limit=15):  # 原先为15
        """按顺序运行所有播出并返回可用的操作及其相应的概率。
        state: 当前游戏的状态
        temp: 介于(0,1]之间的临时参数控制探索的概率
        time_limit: 时间限制,单位为秒
        """
        start_time = time.time()  # 记录开始时间
        self.time_flag=0  # 时间限制
        self.tss_time = 0  # tss调用次数
        for n in range(self._n_playout):
            self.playout_time=n  # 搜索次数
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
            # 检查是否超过了时间限制
            if self.time_flag==0:
                elapsed_time = time.time() - start_time
                if elapsed_time > time_limit:
                    self.time_flag=1  # 超时
                    print("超过时间限制:  ",elapsed_time)
                    print("搜索次数:", n)
            if self.time_flag==1 and n>=700:  # 如果超时但还没搜够700次，继续搜；只有两者都满足才break？？？
                print("搜索次数:", n)
                print("总共花费时间: ",time.time() - start_time)
                break

        # 根据根节点处的访问计数来计算移动概率
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        
        if not act_visits:
            # 如果没有可用的动作,则重新调用get_move_probs
            print("警告：没有可用的动作")
            return self.get_move_probs(state, temp, time_limit)

        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def get_win_rate(self):
        return self._root._Q

    def update_with_move(self, last_move):
        """在当前的树上向前一步,保持我们已经知道的关于子树的一切.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTSPlayer(object):
    """基于MCTS的AI玩家"""

    def __init__(self, policy_value_utss_function, c_puct=5, n_playout=2000, is_selfplay=False):

        self.mcts = MCTS(policy_value_utss_function, c_puct, n_playout, selfplay=is_selfplay)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # 像alphaGo Zero论文一样使用MCTS算法返回的pi向量
        move_probs = np.zeros(board.width*board.height)
        # 当棋盘上的可用位置少于210时,调用tss辅助搜索
        tss_flag = -1
        value_flag = 0
        wise_flag = 0
        if len(sensible_moves) >0:
            if len(sensible_moves)<223:
                print("调用tss辅助搜索")
                current_state = str(board.serialize_board())
                if current_state in board.hash_table_manager.hash_table.keys():
                    print("在必胜hash表中进行查询")
                    move = board.hash_table_manager.hash_table[current_state]
                    print("在必胜hash表中找到了最优点:" + str(board.move_to_location(move)))
                    move_probs[move] = 1
                    tss_flag = 0
                else:
                    board.externalProgramManager.set_board(board.serialize_board())
                    result = board.externalProgramManager.tss(5)

                    if result is not None and result[0] == 1:  # 必胜
                        print("出现必胜点:" + str(board.move_to_location(result[1])))
                        move = result[1]
                        move_probs[move] = 1
                        tss_flag = 0
                        board.hash_table_manager.add(current_state, move)
                        print("将当前局面和必胜点添加到必胜哈希表中")
                    elif result is not None and result[0] == -2:  # 超时
                        tss_flag = 3
                        board.hash_table_manager.add_limited(current_state, -2)
                    elif result is not None and result[0] == -1:  # 必败
                        tss_flag = 2
                        board.hash_table_manager.add_loss(current_state, -1)
                        print("发现局面必败，将当前局面添加到必败表中")
                    elif result is not None and result[0] == 0:  # 平局
                        tss_flag = 1
                        board.hash_table_manager.add_check(current_state, 0)

            if tss_flag != 0:
                if len(sensible_moves)<223:
                    print("tss未能找到最优点,调用MCTS搜索")

                # 获取有价值的落子点
                board.externalProgramManager.set_board(board.serialize_board())
                wise_moves = board.externalProgramManager.wise()

                if len(wise_moves)==1:
                    move = wise_moves[0]
                    move_probs[move] = 1
                    wise_flag = 1
                    print("明智的落子点:" + str(board.move_to_location(move)))
                else:
                    meaning_move = get_meaningmove(board)
                    acts, probs = self.mcts.get_move_probs(board, temp)
                    move_probs = np.zeros(board.width * board.height)  # 重新初始化 move_probs
                    move_probs[list(acts)] = probs
                    if wise_moves:
                        # 如果有价值的落子点不为空,则只在这些点中进行搜索
                        valuable_acts = [act for act in acts if (act in wise_moves and act in meaning_move)]
                        if valuable_acts:
                            value_flag = 1
                            valuable_probs = np.zeros(len(acts))
                            mask = np.array([act in valuable_acts for act in acts])  # 创建一个布尔掩码数组
                            valuable_probs[mask] = probs[mask]
                            valuable_probs_sum = np.sum(valuable_probs)
                            if valuable_probs_sum > 0:  # 检查 valuable_probs 的和是否大于0
                                valuable_probs /= valuable_probs_sum  # 重新归一化概率
                            else:
                                valuable_probs[:] = 1.0 / len(valuable_probs)  # 如果和为0,则将概率均分
                            acts = [act for act in acts if act in valuable_acts]  # 更新 acts
                            probs = valuable_probs[mask]  # 更新 probs
                            move_probs = np.zeros(board.width * board.height)  # 重新初始化 move_probs
                            move_probs[acts] = probs  # 将更新后的 probs 赋值给 move_probs 对应的位置

                            print("有价值的点为")
                            for i in acts:
                                print(board.move_to_location(i), end=" ")

                            # print()

                            # print(move_probs)

                    if value_flag != 1:

                        print("可行的点为")
                        for i in meaning_move:
                            print(board.move_to_location(i), end=" ")
                        print()
                        wise_probs = np.zeros(len(acts))
                        mask = np.array([act in meaning_move for act in acts])  # 创建一个布尔掩码数组
                        wise_probs[mask] = probs[mask]
                        wise_probs_sum = np.sum(wise_probs)
                        if wise_probs_sum > 0:  # 检查 valuable_probs 的和是否大于0
                            wise_probs /= wise_probs_sum  # 重新归一化概率
                        else:
                            wise_probs[:] = 1.0 / len(wise_probs)  # 如果和为0,则将概率均分
                        acts = [act for act in acts if act in meaning_move]  # 更新 acts
                        probs = wise_probs[mask]  # 更新 probs
                        move_probs = np.zeros(board.width * board.height)  # 重新初始化 move_probs
                        move_probs[acts] = probs  # 将更新后的 probs 赋值给 move_probs 对应的位置
                        print(move_probs)

            if self._is_selfplay:
                if tss_flag!=0 and wise_flag==0:
                            p = 0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                            # 添加Dirichlet Noise进行探索(自我训练所需)
                            move = np.random.choice(acts, p=p)
                else:
                    move = move
                # 更新根节点并重用搜索树
                self.mcts.update_with_move(move)
            else:
                if tss_flag!=0 and wise_flag==0:

                    probs_sum = np.sum(probs)
                    if probs_sum > 0:
                        probs = probs / probs_sum
                    else:
                        probs[:] = 1.0 / len(probs)
                    # 使用默认的temp = 1e-3,它几乎相当于选择具有最高概率的移动
                    move = np.random.choice(acts, p=probs)
                else :
                    move=move
                # 重置根节点
                self.mcts.update_with_move(-1)
            if return_prob:
                # 返回动作和动作的概率
                # 打印move以及其对应的概率
                # 返回动作和动作的概率
                # 找到最大概率的动作及其概率值
                max_prob_move = np.argmax(move_probs)
                max_prob = move_probs[max_prob_move]
                max_prob_location = board.move_to_location(max_prob_move)

                print("Max Probability Move: ", max_prob_location, " Probability: ", max_prob)
                location = board.move_to_location(move)
                print("Move: ", location, " Probability: ", move_probs[move])

                return move, move_probs, tss_flag
            else:
                return move
        else:
            print("棋盘已满")

    def __str__(self):
        # return "MCTS {}".format(self.player)

        return "MCTS_player"

# 走法过滤
def get_meaningmove(board):
    res=[]
    for pos in board.states.keys():
        for i in range(-3, 4):
            for j in range(-3, 4):
                    new_pos = pos + i * board.width + j
                    if 0 <= new_pos < board.width * board.height and new_pos not in res and new_pos not in board.states.keys() and new_pos in board.availables:
                        res.append(new_pos)
    return res