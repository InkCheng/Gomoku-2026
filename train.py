# !/usr/bin/env python
# -*- coding: utf-8 -*-
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
from src.players import AIplayer
from src.models.tss_classifier_old import TSSClassifier
import torch
from src.game import Board, Game, HashTableManager
from src.mcts.mcts_alphazero2 import MCTSPlayer as MCTSPlayer2
from src.mcts.mcts_alphazero1 import MCTSPlayer as MCTSPlayer1
from src.models.policy_value_net import PolicyValueNet
from src.models.policy_value_utss_net import PolicyValueUTSSNet
from datetime import datetime
from torch import nn


class TrainPipeline():
    def __init__(self, init_model=None, tss_pretrain=None, is_shown=False, ExternalProgramManager=None, hash_table_manager=None):
        self.loss_log_path = os.path.join(os.getcwd(), 'threeHead_loss_log_new.txt')
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
        # 训练参数
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # 基于KL自适应地调整学习率
        self.temp = 1.0  # 临时变量
        self.n_playout = 500  # 每次移动的模拟次数
        self.c_puct = 2  # 探索参数
        self.buffer_size = 20000  # 经验池大小 10000
        self.batch_size = 1024  # 训练的mini-batch大小 1024
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # 每次更新的train_steps数量
        self.kl_targ = 0.02
        self.check_freq = 50  # 评估模型的频率，可以设置大一些比如500
        self.game_batch_num = 1000000000
        self.best_win_ratio = 0.0
        # 用于纯粹的mcts的模拟数量，用作评估训练策略的对手
        self.pure_mcts_playout_num = 1500
        self.recent_data_ratio = 0.5
        self.old_data_ratio = 0.5
        self.min_batch_size = 256
        self.quick_start_epochs = 5
        self.current_step = 0
        self.tss_pretrain = tss_pretrain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if init_model:
            # 从初始的策略价值utss网络开始训练
            self.policy_value_utss_net = PolicyValueUTSSNet(self.board_width,
                                                            self.board_height,
                                                            model_file=init_model)
            self.tss_classifier = TSSClassifier(num_channels=128, num_res_blocks=7).to(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.tss_classifier.load_state_dict(torch.load('best_model/tss_classifier_800.pth'))
        else:
            # 从新的策略价值utss网络开始训练
            self.policy_value_utss_net = PolicyValueUTSSNet(self.board_width, self.board_height)
            self.tss_classifier = TSSClassifier(num_channels=128, num_res_blocks=7).to(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.tss_classifier.load_state_dict(torch.load('best_model/tss_classifier_800.pth'))

        # 定义训练机器人
        self.mcts_player = MCTSPlayer2(self.policy_value_utss_net.policy_value_utss_fn,
                                       c_puct=self.c_puct,
                                       n_playout=self.n_playout,
                                       is_selfplay=True)

    def load_pretrained_tss_head(self, path):
        print(f"Loading pretrained TSS head from {path}")
        net = self.policy_value_utss_net.policy_value_utss_net
        model_dict = net.state_dict()
        pretrained = torch.load(path, map_location=self.device)
        tss_only = {k: v for k, v in pretrained.items() if
                    'utss' in k and k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(tss_only)
        net.load_state_dict(model_dict)
        print(f"✓ Loaded {len(tss_only)} TSS head parameters.")


    def get_equi_data(self, play_data):
        """通过旋转和翻转来增加数据集
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, tss_flag, winner in play_data:
            state = np.array(state)
            for i in [1, 2, 3, 4]:
                # 逆时针旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    tss_flag,
                                    winner))
                # 水平翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    tss_flag,
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """收集自我博弈数据进行训练"""
        for i in range(n_games):
            # 训练机器人自我对弈
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)  # winner, [states, mcts_probs, winners_z(自己能不能赢)]
            play_data = list(play_data)[:]  # [states, mcts_probs, tss_flags, winners_z(自己能不能赢)]
            # 将对弈数据保存到文件中
            self.episode_len = len(play_data)
            # 增加数据
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

        return play_data


    def stratified_sample(self, batch_size):
        buffer_size = len(self.data_buffer)  # 目前经验回放池大小
        recent_data_size = int(buffer_size * 0.3)  # 最近的数据大小：目前经验回放池大小 * 0.3

        recent_sample_size = int(batch_size * self.recent_data_ratio)  # mini-batch最近的样本数大小：1024 * 0.5
        old_sample_size = batch_size - recent_sample_size  # mini-batch旧样本数大小：1024 - 1024 * 0.5

        recent_data = list(self.data_buffer)[-recent_data_size:]  # 经验回放池中最近的数据
        old_data = list(self.data_buffer)[:-recent_data_size]  # 经验回放池中的旧数据

        if len(recent_data) < recent_sample_size:  # 如果经验回放池中最近的数据量比mini-batch最近的样本数大小还小
            additional_old_sample_size = recent_sample_size - len(
                recent_data)  # 额外的旧样本数大小 = mini-batch最近的样本数大小 - 经验回放池最近的数据大小
            recent_sample = recent_data  # 拷贝经验回放池最近的数据
            old_sample = random.sample(old_data, old_sample_size + additional_old_sample_size)
        else:
            recent_sample = random.sample(recent_data, recent_sample_size)
            old_sample = random.sample(old_data, old_sample_size)

        return recent_sample + old_sample

    def policy_update(self):
        mini_batch = self.stratified_sample(self.batch_size)
        state_batch = np.array([data[0] for data in mini_batch]).astype("float32")
        mcts_probs_batch = np.array([data[1] for data in mini_batch]).astype("float32")
        utss_batch = np.array([data[2] for data in mini_batch]).astype("float32")
        winner_batch = np.array([data[3] for data in mini_batch]).astype("float32")

        # 记录旧策略和价值输出，用于计算 KL 散度
        old_log_probs, old_v, _ = self.policy_value_utss_net.policy_value_utss(state_batch, use_tss=False)
        old_probs = np.exp(old_log_probs)

        total_loss = kl = 0
        for i in range(self.epochs):
            policy_loss, value_loss, utss_loss, vtss_loss, total_loss, entropy = self.policy_value_utss_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                utss_batch,
                self.learn_rate * self.lr_multiplier)

            # 重新预测新策略，计算KL散度
            new_log_probs, new_v, _ = self.policy_value_utss_net.policy_value_utss(state_batch, use_tss=False)
            new_probs = np.exp(new_log_probs)

            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1)
                         )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly kl散度很差提前终止
                print(f"Early stopping at epoch {i + 1} due to high KL divergence: {kl:.5f}")
                break
        # 自适应调节学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print((
            f"kl: {kl:.5f}, "
            f"lr_multiplier: {self.lr_multiplier:.3f}, "
            f"loss: {total_loss}, "
            f"entropy: {entropy}, "
            f"explained_var_old: {explained_var_old:.3f}, "
            f"explained_var_new: {explained_var_new:.3f}"
        ))

        return policy_loss, value_loss, utss_loss, vtss_loss, total_loss, entropy

    def policy_evaluate(self, n_games=4):
        """
        通过与纯的MCTS算法对抗来评估训练的策略
        注意：这仅用于监控训练进度
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
        aiplayer = AIplayer(model_path="best_model/current_policy_step_best.model")
        mcts_player_best = aiplayer.mcts_player
        # policy_value_net_best = PolicyValueNet(15, 15, model_file='best_model/current_policy_step_best.model')
        # mcts_player_best = MCTSPlayer1(policy_value_net_best.policy_value_fn, self.c_puct, self.n_playout, is_selfplay=0, tss_classifier=self.tss_classifier)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            print(f"pk in {i + 1} 局")
            winner = self.game.start_play(current_mcts_player,
                                          mcts_player_best,
                                          start_player=i % 2)
            win_cnt[winner] += 1

        win_ratio = 1.0 * (win_cnt[0] + 0.5 * win_cnt[-1]) / n_games
        print("win: {}, lose: {}, tie:{}".format(win_cnt[0], win_cnt[1], win_cnt[-1]))

        if win_ratio >= 0.75:
            print("新的最好模型产生")
            current_time = datetime.now().strftime('%Y%m%d%H%M%S')  # 使用纯数字格式的时间戳
            best_model_name_now = f"{current_time}_best_current_threeHead_step+{win_ratio:.2f}.model"
            self.policy_value_utss_net.save_model(os.path.join(self.dst_path, best_model_name_now))
            self.policy_value_utss_net.save_model(os.path.join(self.dst_path, 'current_threeHead_step_best.model'))
        return win_ratio

    def run(self):
        num = 0
        """开始训练"""
        root = os.getcwd()

        self.dst_path = os.path.join(root, 'model')

        if not os.path.exists(self.dst_path):
            os.makedirs(self.dst_path)

        try:
            # 开始自博弈学习
            for i in range(self.game_batch_num):
                self.current_step = i
                # 条件式加载 TSS 头（只执行一次）
                if self.current_step == 2 and self.tss_pretrain:
                    self.load_pretrained_tss_head(self.tss_pretrain)
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}".format(i + 1))
                if len(self.data_buffer) > self.batch_size:
                    policy_loss, value_loss, utss_loss, vtss_loss, total_loss, entropy = self.policy_update()
                    print("loss :{}, entropy:{}".format(total_loss, entropy))
                    # 将损失保存到文件
                    with open(self.loss_log_path, 'a', encoding='utf-8') as f:
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        log_line = (
                            f"Epoch {i + 1}, 时间 {current_time}, "
                            f"策略损失: {policy_loss:.4f}, "
                            f"价值损失: {value_loss:.4f}, "
                            f"TSS损失: {utss_loss:.4f}, "
                            f"价值与TSS约束损失: {utss_loss:.4f}, "
                            f"总损失: {total_loss:.4f}\n"
                        )
                        f.write(log_line)

                if (i + 1) % self.check_freq == 0:
                    self.policy_value_utss_net.save_model(
                        os.path.join(self.dst_path, 'current_threeHead.model' + str(i + 1)))
                    print("current self-play batch: {}".format(i + 1))
                    self.game.is_shown = True  # 评估时显示界面
                    self.policy_evaluate()


        except KeyboardInterrupt:
            self.policy_value_utss_net.save_model(os.path.join(self.dst_path, 'current_threeHead_step_temp_quit.model'))
            print("model saved")
            print('\n\rquit')


if __name__ == '__main__':
    externalProgramManager = gomoku_engine.Board()
    hash_table_manager = HashTableManager("merged_hash_new2.pkl")
    model_path = "model/current_threeHead.model400"
    # model_path = None
    tss_pretrain = "model/utss_best1.pth"
    training_pipeline = TrainPipeline(model_path, tss_pretrain, is_shown=True, ExternalProgramManager=externalProgramManager,
                                      hash_table_manager=hash_table_manager)  # shown仅控制训练时是否可视化   平谷时一定可视化
    training_pipeline.run()
    externalProgramManager.terminate()
