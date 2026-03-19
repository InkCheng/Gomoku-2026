import os
import pickle
from src.models.tss_classifier_old import TSSClassifier
from tqdm import tqdm

from src.models.policy_value_utss_net import PolicyValueUTSSNet
from src.game import Game, Board
import torch
import numpy as np
from src.mcts.MCTS_alphazero_test import MCTSPlayer
from src.mcts.mcts_alphazero1 import MCTSPlayer as MCTSPlayer_MC

class AIplayer:
    def __init__(self, model_path="best_model/current_policy_step_best.model",c_puct=2, n_playout=800, is_selfplay=False):
        print("正在加载 AI 模型...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        if torch.cuda.is_available():
            print(f"GPU 名称: {torch.cuda.get_device_name(0)}")

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.to(device)

        self.tss_classifier = TSSClassifier(num_channels=128, num_res_blocks=7).to(device)
        print(f"TSS Classifier 设备: {self.tss_classifier.device}")

        self.tss_classifier.load_state_dict(torch.load('best_model/tss_classifier_800.pth', map_location=device))
        self.tss_classifier.eval()
        self.mcts_player = MCTSPlayer(c_puct, n_playout, is_selfplay,model=self.model,tss_classifier=self.tss_classifier)
        print("AI 模型加载完成！")
        


    def get_action(self, board, temp=1e-3, return_prob=0):
        # 使用MCTS获取动作
        if return_prob:
            move, move_probs = self.mcts_player.get_action(board, temp, return_prob)
            return move,move_probs
        else:
            move=self.mcts_player.get_action(board, temp, return_prob)
            return move

    def evaluate(self, state):
        # 使用模型评估局面
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

        return list(enumerate(act_probs)), value.item()

    def set_player_ind(self, p):
        self.player = p


class Human:

    def __init__(self, width, height):
        self.agent = 'HUMAN'
        self.width = width
        self.height = height

    def get_action(self, move):
        x, y =int(str(input('x= '))), int(str(input('y= ')))
        move = x*self.width+y
        return move

    def set_player_ind(self, p):
        self.player = p


class AIPlayer_MCTS:
    def __init__(self, model_path="D:\\New_Gomoku\\connect5_for_comptition - 副本 - 副本\\best_model\\current_policy.model4200",c_puct=2, n_playout=800, is_selfplay=False):
        self.policy_value_net_now = PolicyValueUTSSNet(15,
                                              15,
                                              model_file=model_path)
        self.tss_classifier = TSSClassifier(num_channels=128, num_res_blocks=7).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.tss_classifier.load_state_dict(torch.load('model/tss_classifier_800.pth'))
        self.tss_classifier.eval()
        self.mcts_player_now = MCTSPlayer_MC(self.policy_value_net_now.policy_value_fn, c_puct, n_playout, 0,self.tss_classifier)

    def get_action(self, board, temp=1e-3, return_prob=0):
        # 使用MCTS获取动作
        if return_prob:
            move, move_probs = self.mcts_player_now.get_action(board, temp, return_prob)
            return move,move_probs
        else:
            move=self.mcts_player_now.get_action(board, temp, return_prob)
            return move

    def evaluate(self, state):
            action_probs, leaf_value = self.policy_value_net_now.policy_value_fn(state)
            return action_probs, leaf_value


# ─── 并行 AI 玩家 (NNBatchServer + MCTSParallel) ───────────────

class AIPlayerParallel:
    """
    并行 AI 玩家: NNBatchServer 批量推理 + 多线程 MCTS。
    接口与 AIplayer 完全一致。
    """

    def __init__(self, model_path="best_model/current_policy_step_best.model",
                 c_puct=2, n_playout=800, is_selfplay=False,
                 num_workers=4, max_batch_size=8, batch_timeout=0.001):
        from src.nn_batch_server import NNBatchServer
        from src.mcts.mcts_parallel import MCTSPlayerParallel

        print("[AIPlayerParallel] 正在加载 AI 模型...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[AIPlayerParallel] 使用设备: {device}")

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.to(device)

        self.tss_classifier = TSSClassifier(
            num_channels=128, num_res_blocks=7
        ).to(device)
        self.tss_classifier.load_state_dict(
            torch.load('best_model/tss_classifier_800.pth',
                        map_location=device)
        )
        self.tss_classifier.eval()

        # 启动 NN 批处理服务
        self.nn_server = NNBatchServer(
            model=self.model,
            tss_classifier=self.tss_classifier,
            max_batch_size=max_batch_size,
            batch_timeout=batch_timeout,
            device=device,
        )
        self.nn_server.start()

        # 创建并行 MCTS 玩家
        self.mcts_player = MCTSPlayerParallel(
            c_puct=c_puct,
            n_playout=n_playout,
            is_selfplay=is_selfplay,
            model=self.model,
            tss_classifier=self.tss_classifier,
            nn_batch_server=self.nn_server,
            num_workers=num_workers,
        )
        print("[AIPlayerParallel] AI 模型加载完成！"
              f" ({num_workers} workers, batch_size={max_batch_size})")

    def get_action(self, board, temp=1e-3, return_prob=0):
        if return_prob:
            move, move_probs = self.mcts_player.get_action(
                board, temp, return_prob
            )
            return move, move_probs
        else:
            return self.mcts_player.get_action(board, temp, return_prob)

    def evaluate(self, state):
        state_tensor = torch.from_numpy(
            state.model_current_state()
        ).cuda().unsqueeze(0).float()
        log_act_probs, value = self.model(state_tensor)
        act_probs = np.exp(log_act_probs.detach().cpu().numpy().flatten())
        available_actions = state.availables
        for i in range(len(act_probs)):
            if i not in available_actions:
                act_probs[i] = 0
        act_probs /= np.sum(act_probs)
        return list(enumerate(act_probs)), value.item()

    def set_player_ind(self, p):
        self.player = p

    def shutdown(self):
        """关闭 NN 批处理服务"""
        if hasattr(self, 'nn_server'):
            self.nn_server.shutdown()