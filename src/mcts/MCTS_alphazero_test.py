
import numpy as np
import copy

import torch
import time

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """MCTS树中的节点。

    每个节点跟踪其自身的值Q，先验概率P及其访问次数调整的先前得分u。
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # 从动作到TreeNode的映射
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self._tss_checked = False  # 新增属性，标记是否已进行过TSS检查


    def expand(self, action_priors, board):
        """通过创建新子项来展开树。
        action_priors：一系列动作元组及其先验概率根据策略函数.
        board: 当前的棋盘状态
        """
        board.externalProgramManager.set_board(board.serialize_board())
        value_flag,valuable_moves = board.externalProgramManager.valuable()
        if value_flag:
            if valuable_moves:
                for action, prob in action_priors:
                    if action in valuable_moves and action not in self._children:
                        self._children[action] = TreeNode(self, prob)

        if not value_flag or not valuable_moves:
            wise_move = get_wisemove(board)
            if wise_move:
                for action, prob in action_priors:
                    if action in wise_move and action not in self._children:
                        self._children[action] = TreeNode(self, prob)
            else:
                # 如果没有wise_move,则按原来的方式展开所有动作
                for action, prob in action_priors:
                    if action not in self._children:
                        self._children[action] = TreeNode(self, prob)

        if self._children=={}:
            #警告
            print("警告：self._children所有动作的概率都是0")
             # 如果没有wise_move,则按原来的方式展开所有动作
            for action, prob in action_priors:
                if action not in self._children:
                    self._children[action] = TreeNode(self, prob)


    def select(self, c_puct,board):
        """在子节点中选择能够提供最大行动价值Q的行动加上奖金u（P）。
     return：（action，next_node）的元组
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
        """就像调用update（）一样，但是对所有祖先进行递归应用。
        """
        # 如果它不是根节点，则应首先更新此节点的父节点。
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """计算并返回此节点的值。它是叶评估Q和此节点的先验的组合
     调整了访问次数，u。
     c_puct：控制相对影响的（0，inf）中的数字，该节点得分的值Q和先验概率P.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """检查叶节点（即没有扩展的节点）。"""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """对蒙特卡罗树搜索的一个简单实现"""

    def __init__(self,  c_puct=5, n_playout=10000,model=None,tss_classifier=None):
        """
        policy_value_fn：一个接收板状态和输出的函数（动作，概率）元组列表以及[-1,1]中的分数
             （即来自当前的最终比赛得分的预期值玩家的观点）对于当前的玩家。
    c_puct：（0，inf）中的数字，用于控制探索的速度收敛于最大值政策。 更高的价值意味着
             依靠先前的更多。
        """
        self.model=model
        self._root = TreeNode(None, 1.0)
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.N_base = 100  # 触发TSS检查的基础访问阈值
        self.K = 0.1  # 触发TSS检查的相对频率阈值
        self.tss_classifier = tss_classifier
        if self.tss_classifier:
            self.device = next(self.tss_classifier.parameters()).device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_leaf(self, state):
        # 使用模型评估叶子节点
        state_tensor = torch.from_numpy(state.model_current_state()).cuda().unsqueeze(0).float()
        log_act_probs, value = self.model(state_tensor)
        act_probs = np.exp(log_act_probs.detach().cpu().numpy().flatten())

        # 将违规点的落子概率设置为0
        available_actions = state.availables
        for i in range(len(act_probs)):
            if i not in available_actions:
                act_probs[i] = 0

        # 重新归一化概率
        act_probs /= np.sum(act_probs)

        # 如果都是0，则报错处理
        if np.sum(act_probs) == 0:
            print("警告：所有动作的概率都是0")

        return list(enumerate(act_probs)), value.item()


    def _playout(self, state):
        """从根到叶子运行单个播出，获取值
         叶子并通过它的父母传播回来。
         State已就地修改，因此必须提供副本。
        """
        node = self._root
        depth = 0
        while (1):
            if node.is_leaf():
                break

            # 贪心算法选择下一步行动
            action, node = node.select(self._c_puct,state)
            state.do_move(action)
            # 在选择前检查是否应该应用TSS


            depth += 1

        flag = 0
        current_state = str(state.serialize_board())
        if current_state in state.hash_table_manager.hash_table.keys():
            flag = 1
            # # 使用数据增强来收集TSS训练数据
            # augmented_data = get_equi_data_tss(state.current_state(), flag)
            # for aug_state, aug_flag in augmented_data:
            #     self.tss_classifier.collect_training_data(aug_state, aug_flag)
        else:
            if self.time_flag == 0:
                if current_state not in state.hash_table_manager.check_table.keys():
                    if depth <= 4 and state.turn >= 10:
                        board_tensor = torch.from_numpy(np.array(state.current_state())).float().unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            tss_prob = torch.sigmoid(self.tss_classifier(board_tensor)).item()
                        if tss_prob > 0.6:  # 如果tss_classifier认为是必胜点的概率大于0.6
                            print(f"TSS分类器检测到可能的必胜点，概率为{tss_prob:.2f}")
                            print("深度为" + str(depth) + "时,进行tss搜索")
                            print(self.tss_time)
                            self.tss_time += 1
                            state.externalProgramManager.set_board(state.serialize_board())
                            result=state.externalProgramManager.tss(5)

                            if result is not None and result[0] == 1:
                                print("出现必胜点:" + str(state.move_to_location(result[1])))
                                flag = 1
                                # 将当前局面和必胜点添加到哈希表中
                                state.hash_table_manager.add(current_state, result[1])
                                print("将当前局面和必胜点添加到哈希表中")
                            elif result is not None and result[0] == -2:
                                state.hash_table_manager.add_limited(current_state, -2)
                                state.hash_table_manager.add_check(current_state, -1)
                            elif result is not None and result[0] == -1:
                                state.hash_table_manager.add_check(current_state, -1)

        if flag == 1:
            end = 1
            winner = state.get_current_player()
        # 查看游戏是否结束
        else:
            end, winner = state.game_end()

        if not end:
            action_probs, leaf_value = self.evaluate_leaf(state)
            # 如果action_probs都是0，则报错处理
            node.expand(action_probs,state)
        else:
            # 对于结束状态,将叶子节点的值换成"true"
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # 在本次遍历中更新节点的值和访问次数
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3,time_limit=15):

        """按顺序运行所有播出并返回可用的操作及其相应的概率。
        state: 当前游戏的状态
        temp: 介于(0,1]之间的临时参数控制探索的概率
        """

        start_time = time.time()  # 记录开始时间

        self.time_flag = 0
        self.tss_time = 0
        for n in range(self._n_playout):
            self.playout_time = n
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

            # 主动释放 GIL 锁，让 UI 线程（Tkinter主循环）有喘息机会，从而消除界面卡顿和计时器冻结现象
            if n % 10 == 0:
                time.sleep(0.001)

            # 检查是否超过了时间限制
            if self.time_flag == 0:
                elapsed_time = time.time() - start_time
                if elapsed_time > time_limit:
                    self.time_flag = 1
                    print("超过时间限制:  ", elapsed_time)
                    print("搜索次数：", n)
            if self.time_flag == 1 and n >= 600:
                print("搜索次数：", n)
                print("总共花费时间: ", time.time() - start_time)
                break


        # 根据根节点处的访问计数来计算移动概率
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        print(self._root._children)
        print(act_visits)

        if not act_visits:
            print("警告：没有可用的动作")
            return self.get_move_probs(state, temp, time_limit)

        acts, visits = zip(*act_visits)
        # act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        # 获取所有可用的操作
        valid_moves = state.availables

        # 创建一个全为0的概率数组,长度与acts一致
        act_probs = np.zeros(len(acts))

        # 获取有效操作的索引
        valid_indices = [i for i in range(len(acts)) if acts[i] in valid_moves]

        if valid_indices:
            # 获取有效操作的访问次数
            valid_visits = [visits[i] for i in valid_indices]

            # 重新计算有效操作的概率
            valid_probs = softmax(1.0 / temp * np.log(np.array(valid_visits) + 1e-10))

            # 将有效操作的概率赋值到对应的位置
            act_probs[valid_indices] = valid_probs

        return acts, act_probs

    def get_win_rate(self):
        return self._root._Q

    def update_with_move(self, last_move):
        """在当前的树上向前一步，保持我们已经知道的关于子树的一切.
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

    def __init__(self, c_puct=5, n_playout=400, is_selfplay=0,model=None,tss_classifier=None):
        self.model=model
        self.tss_classifier=tss_classifier
        self.mcts = MCTS( c_puct, n_playout,self.model,tss_classifier=self.tss_classifier)
        self._is_selfplay = is_selfplay
        

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # 像alphaGo Zero论文一样使用MCTS算法返回的pi向量
        move_probs = np.zeros(board.width*board.height)
        # 当棋盘上的可用位置少于210时，调用tss辅助搜索
        tss_flag = 0
        flag=0
        if len(sensible_moves) >0:
            if len(sensible_moves)<223:
                print("调用tss辅助搜索")
                current_state = str(board.serialize_board())
                if current_state in board.hash_table_manager.hash_table.keys():
                    print("在hash表中进行查询")
                    move = board.hash_table_manager.hash_table[current_state]
                    print("在hash表中找到了最优点：" + str(board.move_to_location(move)))
                    move_probs[move] = 1
                    tss_flag = 1
                else:
                    board.externalProgramManager.set_board(board.serialize_board())
                    result = board.externalProgramManager.tss(10)
                    
                    if result is not None and result[0] == 1:
                        print("出现必胜点：" + str(board.move_to_location(result[1])))
                        move = result[1]
                        move_probs[move] = 1
                        tss_flag = 1
                        board.hash_table_manager.add(current_state, move)
                        print("将当前局面和必胜点添加到哈希表中")
                    elif result is not None and result[0] == -2:
                        board.hash_table_manager.add_limited(current_state, -2)
                        board.hash_table_manager.add_check(current_state, -1)
                    elif result is not None and result[0] == -1:
                        board.hash_table_manager.add_check(current_state, -1)


            if tss_flag == 0:
                if len(sensible_moves)<223:
                    print("tss未能找到最优点，调用MCTS搜索")

                # 获取有价值的落子点
                board.externalProgramManager.set_board(board.serialize_board())
                value_flag,valuable_moves = board.externalProgramManager.valuable()

                if len(valuable_moves)==1:
                    move = valuable_moves[0]
                    move_probs[move] = 1
                    flag = 1
                    print("有价值的落子点：" + str(board.move_to_location(move)))
                else:
                    time_limit = getattr(self, "search_time", 15)
                    acts, probs = self.mcts.get_move_probs(board, temp, time_limit=time_limit)
                    move_probs = np.zeros(board.width * board.height)  # 重新初始化 move_probs
                    move_probs[list(acts)] = probs
                    wise_move = get_wisemove(board)
                    wise_probs = np.zeros(len(acts))
                    mask = np.array([act in wise_move for act in acts])  # 创建一个布尔掩码数组
                    wise_probs[mask] = probs[mask]
                    wise_probs_sum = np.sum(wise_probs)
                    if wise_probs_sum > 0:  # 检查 valuable_probs 的和是否大于0
                        wise_probs /= wise_probs_sum  # 重新归一化概率
                    else:
                        wise_probs[:] = 1.0 / len(wise_probs)  # 如果和为0,则将概率均分
                    acts = [act for act in acts if act in wise_move]  # 更新 acts
                    probs = wise_probs[mask]  # 更新 probs
                    move_probs = np.zeros(board.width * board.height)  # 重新初始化 move_probs
                    move_probs[acts] = probs  # 将更新后的 probs 赋值给 move_probs 对应的位置

            if self._is_selfplay:
                if tss_flag==0 and flag==0:
                            p = 0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                            # 添加Dirichlet Noise进行探索（自我训练所需）
                            move = np.random.choice(
                                acts,
                                p=p
                         )

                else:
                    move = move
                # 更新根节点并重用搜索树
                self.mcts.update_with_move(move)
            else:
                if tss_flag==0 and flag==0:

                    probs_sum = np.sum(probs)
                    if probs_sum > 0:
                        probs = probs / probs_sum
                    else:
                        probs[:] = 1.0 / len(probs)
                    # 使用默认的temp = 1e-3，它几乎相当于选择具有最高概率的移动
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

                return move, move_probs
            else:
                return move
        else:
            print("棋盘已满")

    def __str__(self):
        # return "MCTS {}".format(self.player)

        return "MCTS_player"

def get_wisemove(board):
    res=[]
    for pos in board.states.keys():
        for i in range(-3, 4):
            for j in range(-3, 4):
                    new_pos = pos + i * board.width + j
                    if 0 <= new_pos < board.width * board.height and new_pos not in res and new_pos not in board.states.keys() and new_pos in board.availables:
                        res.append(new_pos)
    return res