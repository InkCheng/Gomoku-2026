import gomoku_engine

# 导入socket库
import json
import pickle
import socket
#  对于五子棋的AlphaZero的训练的实现
import ast
import random
from multiprocessing import Process
import numpy as np
import os
from collections import defaultdict, deque
from src.models.tss_classifier_old import TSSClassifier
import torch
from src.game import Board, Game,  HashTableManager
from src.mcts.mcts_alphazero2 import MCTSPlayer as MCTSPlayer2
from src.mcts.mcts_alphazero1 import MCTSPlayer as MCTSPlayer1
from src.models.policy_value_net import PolicyValueNet
from src.models.policy_value_utss_net import PolicyValueUTSSNet
from datetime import datetime
from torch import nn


class pk():
    def __init__(self, init_model=None, is_shown=False,ExternalProgramManager=None,hash_table_manager=None):
        # 五子棋逻辑和棋盘UI的参数
        self.board_width = 15
        self.board_height = 15
        self.n_in_row = 5
        self.board = Board(ExternalProgramManager=ExternalProgramManager, width=self.board_width,
                           hash_table_manager=hash_table_manager,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.is_shown = is_shown
        self.game = Game(board=self.board, is_shown=self.is_shown)
        self.best_win_ratio = 0.0
        self.c_puct = 2.0
        self.n_playout = 500

        if init_model:
            # 从初始的策略价值utss网络开始训练
            self.policy_value_utss_net = PolicyValueUTSSNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
            self.tss_classifier = TSSClassifier(num_channels=128, num_res_blocks=7).to('cuda' if torch.cuda.is_available() else 'cpu')
            self.tss_classifier.load_state_dict(torch.load('best_model/tss_classifier_800.pth'))
        else:
            # 从新的策略价值utss网络开始训练
            self.policy_value_utss_net = PolicyValueUTSSNet(self.board_width, self.board_height)
            self.tss_classifier = TSSClassifier(num_channels=128, num_res_blocks=7).to('cuda' if torch.cuda.is_available() else 'cpu')
            self.tss_classifier.load_state_dict(torch.load('best_model/tss_classifier_800.pth'))

        # 定义训练机器人
        self.mcts_player = MCTSPlayer2(self.policy_value_utss_net.policy_value_utss_fn,
                                       c_puct=self.c_puct,
                                       n_playout=self.n_playout,
                                       is_selfplay=True)

    def policy_evaluate(self, n_games=2):
        """
        通过与纯的MCTS算法对抗来评估训练的策略
        """
        best_model_path = 'model/current_threeHead_step_best.model'
        if not os.path.exists(best_model_path):
            self.policy_value_utss_net.save_model(os.path.join(self.dst_path, 'current_threeHead_step_best.model'))
            print(f"模型文件 {best_model_path} 不存在，目前的模型为最好模型")
            return 1.0  # 假设当前模型为最好的模型，返回胜率1.0

        # 当前最新模型玩家（用三头网络）
        current_mcts_player = MCTSPlayer2(self.policy_value_utss_net.policy_value_utss_fn,
                                          c_puct=self.c_puct,
                                          n_playout=self.n_playout,
                                          is_selfplay=False)

        # # 加载历史最佳模型 (用三头网络)
        # policy_value_utss_net_best = PolicyValueUTSSNet(15, 15, model_file='model/current_threeHead_step_best.model')
        # mcts_player_best = MCTSPlayer2(policy_value_utss_net_best.policy_value_utss_fn,
        #                                c_puct=self.c_puct,
        #                                n_playout=self.n_playout,
        #                                is_selfplay=False)

        # 加载历史最佳模型，用PolicyValueNet两头结构+tssclassifier
        policy_value_net_best = PolicyValueNet(15, 15, model_file='best_model/current_policy.model4200')
        mcts_player_best = MCTSPlayer1(policy_value_net_best.policy_value_fn, self.c_puct, self.n_playout, is_selfplay=False, tss_classifier=self.tss_classifier)

        win_cnt = defaultdict(int)
        n_games = 11  # 设置对战局数为100局

        for i in range(n_games):
            print(f"pk 第 {i + 1} 局")
            winner = self.game.start_play(current_mcts_player,
                                          mcts_player_best,
                                          start_player=i % 2)
            win_cnt[winner] += 1

        # 计算胜率（0是current_mcts_player赢，1是mcts_player_best赢，-1是平局）
        win_ratio = 1.0 * (win_cnt[0] + 0.5 * win_cnt[-1]) / n_games

        print("结果统计：")
        print(f"current_mcts_player 胜：{win_cnt[0]} 局")
        print(f"mcts_player_best 胜：{win_cnt[1]} 局")
        print(f"平局：{win_cnt[-1]} 局")
        print(f"胜率：{win_ratio * 100:.2f}%")

        if win_ratio > 0.6:
            print("新的最好模型产生")
            # 保存模型的逻辑可以按需启用
            # current_time = datetime.now().strftime('%Y%m%d%H%M%S')
            # best_model_name_now = current_time + '_best_current_threeHead_step.model'
            # self.policy_value_utss_net.save_model(os.path.join(self.dst_path, best_model_name_now))
            # self.policy_value_utss_net.save_model(os.path.join(self.dst_path, 'current_threeHead_step_best.model'))

        return win_ratio

    def run(self):
        self.game.is_shown = True  # 评估时显示界面
        self.policy_evaluate()


if __name__ == '__main__':
    externalProgramManager = gomoku_engine.Board()
    hash_table_manager = HashTableManager("merged_hash_new.pkl")
    model_path = "model/(0.6)best_current_threeHead_step.model"
    # model_path = None
    pk = pk(model_path, is_shown=False, ExternalProgramManager=externalProgramManager,
                                      hash_table_manager=hash_table_manager)  # shown仅控制训练时是否可视化   平谷时一定可视化
    pk.run()