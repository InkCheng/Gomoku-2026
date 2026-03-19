import copy
import os
# import gomoku_engine
import numpy as np
import torch

from src.mcts.mcts_alphazero2 import MCTSPlayer
import subprocess
import time # 用于计时
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

import pickle
import json

# 定义开局局面
opening_moves = {
    "寒星局": [(7, 7), (7, 8), (7, 9)],
    "溪月局": [(7, 7), (7, 8), (9, 8)],
    "疏星局": [(7, 7), (7, 8), (8, 9)],
    "花月局": [(7, 7), (7, 8), (8, 8)],
    "残月局": [(7, 7), (7, 8), (9, 8)],
    "雨月局": [(7, 7), (7, 8), (7, 9)],
    "金星局": [(7, 7), (7, 8), (8, 9)],
    "松月局": [(7, 7), (7, 8), (6, 7)],
    "丘月局": [(7, 7), (7, 8), (8, 6)],
    "新月局": [(7, 7), (7, 8), (8, 6)],
    "瑞星局": [(7, 7), (7, 8), (7, 5)],
    "山月局": [(7, 7), (7, 8), (6, 5)],
    "游星局": [(7, 7), (7, 8), (8, 6)],
    "长星局": [(7, 7), (8, 8), (9, 9)],
    "峡月局": [(7, 7), (8, 8), (9, 8)],
    "恒星局": [(7, 7), (8, 8), (8, 7)],
    "水月局": [(7, 7), (8, 8), (7, 6)],
    "流星局": [(7, 7), (8, 8), (8, 6)],
    "云月局": [(7, 7), (8, 8), (7, 7)],
    "浦月局": [(7, 7), (8, 8), (8, 6)],
    "岚月局": [(7, 7), (8, 8), (7, 6)],
    "银月局": [(7, 7), (8, 8), (5, 7)],
    "明星局": [(7, 7), (8, 8), (8, 7)],
    "斜月局": [(7, 7), (8, 8), (7, 6)],
    "名月局": [(7, 7), (8, 8), (7, 5)],
    "彗星局": [(7, 7), (8, 8), (8, 9)]
}

# 必胜表、必败表、平局检查表、超时未知哈希表
class HashTableManager:
    def __init__(self, file_path):
        self.hash_table = {}        # 必胜表
        self.check_table = {}       # 平局检查表
        self.limited_time_hash = {} # 超时未知哈希表
        self.loss_table = {}        # 必败表

        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                pass
        self.file_path = file_path

        self.check_file_path = 'check' + file_path  # 检查表文件路径
        if not os.path.exists(self.check_file_path):
            with open(self.check_file_path, 'w') as f:
                pass

        self.limited_file_path = 'limited' + file_path  # 限时表文件路径
        if not os.path.exists(self.limited_file_path):
            with open(self.limited_file_path, 'w') as f:
                pass

        self.loss_file_path = 'loss' + file_path  # 必败表文件路径
        if not os.path.exists(self.loss_file_path):
            with open(self.loss_file_path, 'w') as f:
                pass
        # 检查文件是否存在,如果不存在则创建
        self.load_from_file()

    def load_from_file(self):
        try:
            with open(self.file_path, 'rb') as f:
                while True:
                    try:
                        data = pickle.load(f)
                        self.hash_table.update(data)
                    except EOFError:
                        break
            print("必胜表读取成功！")
        except FileNotFoundError:
            print(f"Error loading hash table from {self.file_path}. Creating a new one.")
            self.hash_table = {}

        try:
            with open(self.loss_file_path, 'rb') as f:
                while True:
                    try:
                        data = pickle.load(f)
                        self.loss_table.update(data)
                    except EOFError:
                        break
            print("必败表读取成功！")
        except FileNotFoundError:
            print(f"Error loading hash table from {self.loss_file_path}. Creating a new one.")
            self.loss_table = {}

        try:
            with open(self.check_file_path, 'rb') as f:
                while True:
                    try:
                        data = pickle.load(f)
                        self.check_table.update(data)
                    except EOFError:
                        break
            print("平局检查表读取成功！")
        except FileNotFoundError:
            print(f"Error loading check table from {self.check_file_path}. Creating a new one.")
            self.check_table = {}

        try:
            with open(self.limited_file_path, 'rb') as f:
                while True:
                    try:
                        data = pickle.load(f)
                        self.limited_time_hash.update(data)
                    except EOFError:
                        break
            print("超时未知哈希表读取成功！")
        except FileNotFoundError:
            print(f"Error loading limited time hash table from {self.limited_file_path}. Creating a new one.")
            self.limited_time_hash = {}

    def add_limited(self, key, value):
        self.limited_time_hash[key] = value
        with open(self.limited_file_path, 'ab') as f:
            pickle.dump({key: value}, f)

    def add(self, key, value):
        self.hash_table[key] = value
        with open(self.file_path, 'ab') as f:
            pickle.dump({key: value}, f)

    def add_loss(self, key, value):
        self.loss_table[key] = value
        with open(self.loss_file_path, 'ab') as f:
            pickle.dump({key: value}, f)

    def remove(self, key):
        if key in self.hash_table:
            del self.hash_table[key]
            self.rewrite_file()

    def get(self, key):
        return self.hash_table.get(key)

    def update(self, key, value):
        self.hash_table[key] = value
        self.rewrite_file()

    def add_check(self, key, value):
        self.check_table[key] = value
        with open(self.check_file_path, 'ab') as f:
            pickle.dump({key: value}, f)

    def get_check(self, key):
        return self.check_table.get(key)

    def remove_check(self, key):
        if key in self.check_table:
            del self.check_table[key]
            self.rewrite_check_file()

    def rewrite_file(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.hash_table, f)

    def rewrite_check_file(self):
        with open(self.check_file_path, 'wb') as f:
            pickle.dump(self.check_table, f)

class Board(object):
    """棋盘游戏逻辑控制"""

    def __init__(self, ExternalProgramManager,AI_turn=0,hash_table_manager=None,**kwargs,):
        self.externalProgramManager=ExternalProgramManager
        self.width = int(kwargs.get('width', 15))  # 棋盘宽度
        self.height = int(kwargs.get('height', 15))  # 棋盘高度
        self.states = {}  # 棋盘状态为一个字典,键: 移动步数,值: 玩家的棋子类型
        self.n_in_row = int(kwargs.get('n_in_row', 5))  # 5个棋子一条线则获胜
        self.players = [0, 1]  # 玩家0,1
        # self.rule = Rule()
        self.start_player = 0  # 先手玩家
        self.current_player = 0  # 当前玩家（默认为0）
        self.pre_action = -1  # 上一次的行动
        self.turn = 0  # 当前回合数
        self.non_compliance = 0 # 有无犯规
        self.directions = [[[0, -1], [0, 1]],  # 竖直搜索
                           [[-1, 0], [1, 0]],  # 水平搜索
                           [[-1, -1], [1, 1]],  # 主对角线搜索
                           [[1, -1], [-1, 1]]]  # 副对角线搜索
        self.AI_turn=AI_turn
        self.hash_table_manager = hash_table_manager


    def init_board(self, start_player=0):
        # 初始化棋盘
        # 当前棋盘的宽高小于5时,抛出异常(因为是五子棋)
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('棋盘的长宽不能少于{}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # 先手玩家
        self.availables = list(range(self.width * self.height))  # 初始化可用的位置列表
        self.states = {}  # 初始化棋盘状态
        self.last_move = -1  # 初始化最后一次的移动位置
        self.start_player= start_player  # 初始化先手玩家
        self.turn = 0  # 初始化回合数
        self.non_compliance = 0 # 有无违规操作


    def __deepcopy__(self, memo):
        # 创建一个新的Board实例，不使用原有的外部程序管理器
        new_board = Board(ExternalProgramManager=self.externalProgramManager, hash_table_manager=self.hash_table_manager,width=self.width, height=self.height, n_in_row=self.n_in_row)
        # 深复制需要的属性
        new_board.states = copy.deepcopy(self.states, memo)
        new_board.players = copy.deepcopy(self.players, memo)
        new_board.start_player = copy.deepcopy(self.start_player)
        new_board.availables= copy.deepcopy(self.availables, memo)
        new_board.current_player = copy.deepcopy(self.current_player)
        new_board.last_move = copy.deepcopy(self.last_move)
        new_board.pre_action= copy.deepcopy(self.pre_action)
        new_board.turn = copy.deepcopy(self.turn)
        new_board.n_in_row=copy.deepcopy(self.n_in_row)
        return new_board

    def game_end(self):
        """检查当前棋局是否结束"""
        self.have_non_compliance=0
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            # 棋局布满,没有赢家
            return True, -1
        return False, -1

    def game_over_who_win(self):
        """
        返回胜利方 WHITE BLACK 或则 还未结束 CONTINUE
        2代表没有结果
        3代表平局
        """
        pre_player = 1 - self.current_player
        if self.turn <= 5:
            return 2
        # 需注意play方法过后 双方会变
        if self.non_compliance == 1:
            print("出现违规操作")
            self.non_compliance=0
            self.have_non_compliance=1

            return 1 - pre_player # 违规操作的一方输,对方获胜

        # 查看黑棋白棋有没有连5子
        if self.connect_5(one_x=self.last_move, color=pre_player):
            return pre_player
        # 棋盘布满了，平局
        if self.turn == 225:
            return 3
        return 2

    def connect_5(self, one_x, color) -> bool:
        """
        检查one_x各方向是否连成5子
        """
        row, col = self.move_to_location(one_x)

        for i in range(4):  # 四个方向
            number = 1
            for j in range(2):  # 方向左右
                flag = True
                row_t, col_t = row, col
                while flag:
                    row_t = row_t + self.directions[i][j][0]
                    col_t = col_t + self.directions[i][j][1]
                    current_index = row_t * self.width + col_t  # 计算当前检查的棋子的索引
                    if 0 <= row_t < self.width and 0 <= col_t < self.height and self.states.get(current_index)  == color:
                        number += 1
                    else:
                        flag = False
            if color == self.start_player and number == 5:  # 黑棋必须严格等于5
                return True
            if color != self.start_player and number >= 5:  # 白棋长连也算赢
                return True
        return False

    def get_available(self,current_play) -> list:
        """
        返回当前方在当前棋盘可以行棋的位置的list
        """
        res = []

        self.externalProgramManager.set_board(self.serialize_board())

        if current_play==self.start_player:
            res = self.externalProgramManager.available()
        else:
            for i in range(self.width * self.height):
                if i not in self.states.keys():
                    res.append(i)
        return res

    def serialize_board(self):
        board_state = [['E' for _ in range(self.width)] for _ in range(self.height)]
        for move, player in self.states.items():
            h, w = self.move_to_location(move)
            board_state[h][w] = 'B' if player == self.players[self.start_player] else 'W'
        board_state_str = [''.join(row) for row in board_state]
        return board_state_str

    def move_to_location(self, move):
        # 根据传入的移动步数返回位置(如:move=2,计算得到坐标为[0,2],即表示在棋盘上左上角横向第三格位置)
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        # 根据传入的位置返回移动值
        # 位置信息必须包含2个值[h,w]
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        # 超出棋盘的值不存在
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """
        从当前玩家的角度返回棋盘状态。
        状态形式：7 * 宽 * 高
        """
        # 使用7个15x15的二值特征平面来描述当前的局面
        # 前两个平面分别表示当前player的棋子位置和对手player的棋子位置，有棋子的位置是1，没棋子的位置是0
        # 第三个平面表示的是当前player是不是先手player，如果是先手player则整个平面全部为1，否则全部为0
        # 第四个平面表示 valuable_moves
        # 第五个平面表示 wise_moves
        # 第六个平面表示 available_moves
        # 第七个平面表示 meaning_moves

        square_state = np.zeros((7, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]  # 获取棋盘状态上属于当前玩家的所有移动值
            move_oppo = moves[players != self.current_player]  # 获取棋盘状态上属于对方玩家的所有移动值
            square_state[0][move_curr // self.width,  # 对第一个特征平面填充值(当前玩家)
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,  # 对第二个特征平面填充值(对方玩家)
                            move_oppo % self.height] = 1.0
        if len(self.states) % 2 == 0:  # 对第三个特征平面填充值,当前玩家是先手,则填充全1,否则为全0
            square_state[2][:, :] = 1.0
        self.externalProgramManager.set_board(self.serialize_board())
        Flag,valuable_moves = self.externalProgramManager.valuable()
        wise_moves = self.externalProgramManager.wise()
        available_moves = self.availables
        meaning_moves = []
        for pos in self.states.keys():
            for i in range(-3, 4):
                for j in range(-3, 4):
                    new_pos = pos + i * self.width + j
                    if 0 <= new_pos < self.width * self.height and new_pos not in meaning_moves and new_pos not in self.states.keys() and new_pos in self.availables:
                        meaning_moves.append(new_pos)
            # 更新第四个特征平面
        for move in valuable_moves:
            square_state[3][move // self.width, move % self.height] = 1.0

            # 更新第五个特征平面
        for move in wise_moves:
            square_state[4][move // self.width, move % self.height] = 1.0

            # 计算并更新第六个特征平面
        for move in meaning_moves:
             square_state[5][move // self.width, move % self.height] = 1.0

            # 更新第七个特征平面
        for move in available_moves:
            square_state[6][move // self.width, move % self.height] = 1.0

        # 将每个平面棋盘状态按行逆序转换(第一行换到最后一行,第二行换到倒数第二行..)
        return square_state[:, ::-1, :].copy()

    def current_state_row(self):
        """
        从当前玩家的角度返回棋盘状态。
    状态形式：4 * 宽 * 高
        """
        # 使用4个15x15的二值特征平面来描述当前的局面
        # 前两个平面分别表示当前player的棋子位置和对手player的棋子位置，有棋子的位置是1，没棋子的位置是0
        # 第三个平面表示对手player最近一步的落子位置，也就是整个平面只有一个位置是1，其余全部是0
        # 第四个平面表示的是当前player是不是先手player，如果是先手player则整个平面全部为1，否则全部为0

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]  # 获取棋盘状态上属于当前玩家的所有移动值
            move_oppo = moves[players != self.current_player]  # 获取棋盘状态上属于对方玩家的所有移动值
            square_state[0][move_curr // self.width,  # 对第一个特征平面填充值(当前玩家)
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,  # 对第二个特征平面填充值(对方玩家)
                            move_oppo % self.height] = 1.0
            # 指出最后一个移动位置
            square_state[2][self.last_move // self.width,  # 对第三个特征平面填充值(对手最近一次的落子位置)
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:  # 对第四个特征平面填充值,当前玩家是先手,则填充全1,否则为全0
            square_state[3][:, :] = 1.0
        # 将每个平面棋盘状态按行逆序转换(第一行换到最后一行,第二行换到倒数第二行..)
        return square_state[:, ::-1, :].copy()

    def model_current_state(self):
        square_state = np.zeros((3, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]  # 获取棋盘状态上属于当前玩家的所有移动值
            move_oppo = moves[players != self.current_player]  # 获取棋盘状态上属于对方玩家的所有移动值
            square_state[0][move_curr // self.width,  # 对第一个特征平面填充值(当前玩家)
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,  # 对第二个特征平面填充值(对方玩家)
                            move_oppo % self.height] = 1.0
        # 指出当前玩家的颜色
        square_state[2][:, :] = -1.0 if self.current_player == self.players[self.start_player] else 1.0
        return square_state.copy()


    def do_move(self, move):
        self.turn += 1
        if move not in self.availables:
            # print("非法移动")
            self.non_compliance = 1
        # 根据移动的数据更新各参数
        self.states[move] = self.current_player  # 将当前的参数存入棋盘状态中
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )  # 改变当前玩家
        self.last_move = move  # 记录最后一次的移动位置
        self.availables=self.get_available(self.current_player)
        if self.non_compliance:
            return False
        return True

    def undo_move(self, x, y):
        """
        撤销落子，接受的参数是该点的X和Y值
        """
        move = self.location_to_move((x, y))
        if move in self.states:
            del self.states[move]  # 从棋盘状态中移除该位置的棋子
            self.availables.append(move)  # 将该位置重新加入可用位置列表
            self.last_move = -1 if not self.states else max(self.states.keys())  # 更新最后一次的移动位置
            self.current_player = (
                self.players[0] if self.current_player == self.players[1]
                else self.players[1]
            )  # 恢复上一个玩家
            self.turn -= 1  # 回合数减一
            self.availables = self.get_available(self.current_player)  # 更新可用位置
        else:
            print("该位置没有棋子，无法撤销")

    def has_a_winner(self):
        winner = self.game_over_who_win()

        if winner==2:
            # 当前都没有赢家,返回False
            return False, -1
        elif winner == 0:# 玩家1
            return True,self.players[0]
        else:
            return True, self.players[1]

    def visual(self):
        print('*' * 50)
        for y in reversed(range(self.height)):
            print(f"{y:2d}:\t", end='')  # 确保y轴标签正确对齐
            for x in range(self.width):
                move = self.location_to_move((x, y))
                cell_content = '-\t'  # 空单元格的默认内容
                if move in self.states:
                    player = self.states[move]
                    if player == self.start_player:
                        cell_content = '\033[0;31;40mB\033[0m\t'  # 先手黑方玩家使用红色
                        if move == self.last_move:
                            cell_content = '\033[0;33;40mB\033[0m\t'  # 最后一步使用黄色
                    elif player != self.start_player:
                        cell_content = '\033[0;32;40mV\033[0m\t'  # 后手白方使用绿色
                        if move == self.last_move:
                            cell_content = '\033[0;33;40mV\033[0m\t'  # 最后一步使用黄色
                print(cell_content, end='')
            print()  # 每行结束后换行
        # 对齐列号的打印方式
        print('       ', end='')  # 调整这里以确保与上面的棋盘对齐
        for x in range(self.width):
            print(f"{x}\t",end='')  # 确保x轴标签正确对齐
        print('\n' + '*' * 50)

    def get_current_player(self):
        return self.current_player

    def is_symmetric(self):
        # 判断棋盘状态是否轴对称
        # 返回一个列表,包含四个布尔值,分别表示是否关于横、竖、斜上、斜下轴对称
        symmetric_flags = [True] * 4

        for move, player in self.states.items():
            # 获取落子点的坐标
            row, col = self.move_to_location(move)

            # 检查水平轴对称
            mirror_move = self.location_to_move((self.height - 1 - row, col))
            if self.states.get(move) != self.states.get(mirror_move):
                symmetric_flags[0] = False

            # 检查垂直轴对称
            mirror_move = self.location_to_move((row, self.width - 1 - col))
            if self.states.get(move) != self.states.get(mirror_move):
                symmetric_flags[1] = False

            # 检查斜上轴对称
            mirror_move = self.location_to_move((col, row))
            if self.states.get(move) != self.states.get(mirror_move):
                symmetric_flags[2] = False

            # 检查斜下轴对称
            mirror_move = self.location_to_move((self.width - 1 - col, self.height - 1 - row))
            if self.states.get(move) != self.states.get(mirror_move):
                symmetric_flags[3] = False

        return symmetric_flags


    # def perpendicular_bisector( self, x1, y1, x2, y2):
    #     # 计算中点
    #     mid_x = (x1 + x2) / 2
    #     mid_y = (y1 + y2) / 2
    #
    #     # 计算原始线段的斜率
    #     if x2 - x1 != 0:
    #         m_original = (y2 - y1) / (x2 - x1)
    #         # 垂直平分线的斜率是原始斜率的负倒数
    #         m_perpendicular = -1 / m_original
    #     else:
    #         # 如果原始线段垂直,则垂直平分线水平
    #         m_perpendicular = 0
    #
    #     # 计算y轴截距
    #     b = mid_y - m_perpendicular * mid_x
    #
    #     return m_perpendicular, b

class Game():
    def __init__(self, board: Board, is_shown=False):
        self.board = board
        self.is_shown = is_shown
        self.player1_times = []
        self.player2_times = []


    def start_self_play(self, player, temp=1e-3):
        """
        使用MCTS玩家开始自己玩游戏,重新使用搜索树并存储自己玩游戏的数据
        (state, mcts_probs, z) 提供训练
        :param player:
        :param temp:
        :return:
        """
        self.board.init_board()  # 初始化棋盘
        # 疏影局训练
        self.board.do_move(112)
        self.board.do_move(113)
        self.board.do_move(9 * 15 + 9)
        print("start_self_play")

        states, mcts_probs, current_players, tss_flags = [], [], [], []  # 状态,mcts的行为概率,当前玩家, tss局面

        if self.is_shown:
            self.board.visual()

        while True:
            move, move_probs, tss_flag = player.get_action(self.board,temp=temp,return_prob=1)
            # 存储数据
            states.append(self.board.current_state())  # 存储状态数据
            mcts_probs.append(move_probs)  # 存储行为概率数据
            tss_flags.append(tss_flag)  # 存储tss_flag
            current_players.append(self.board.current_player)  # 存储当前玩家
            # 执行一个移动
            self.board.do_move(move)
            if self.is_shown:
                self.board.visual()

            # 判断该局游戏是否终止
            end, winner = self.board.game_end()
            if end:
                # 从每个状态的当时的玩家的角度看待赢家
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    # 有赢家时
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # 重置MSCT的根节点
                player.reset_player()
                if self.is_shown:
                    if winner == self.board.start_player:
                        print('胜利者 : 黑方（先手） ')
                    else:
                        print('胜利者 : 白方（后手）')
                    print(f"winner : \t{winner}")
                return winner, zip(states, mcts_probs, tss_flags, winners_z)  # 返回的是winner, [states, mcts_probs, tss_flags, winners_z(自己能不能赢)]

    def start_play(self, player1, player2, i, j, start_player=0):
        """开始一局游戏"""
        if start_player not in (0, 1):
            # 如果玩家不在玩家1,玩家2之间,抛出异常
            raise Exception('开始的玩家必须为0(玩家1)或1(玩家2)')

        self.board.init_board(start_player)  # 初始化棋盘
        p1, p2 = self.board.players  # 加载玩家1,玩家2
        player1.set_player_ind(p1)  # 设置玩家1
        player2.set_player_ind(p2)  # 设置玩家2
        players = {p1: player1, p2: player2}
        self.board.do_move(112)
        self.board.do_move(113)
        self.board.do_move(9 * 15 + 9)
        self.player1_id = i
        self.player2_id = j
        if self.is_shown:
            self.board.visual()
        #轮换开局模式
        # opening_cycle = itertools.cycle(opening_moves.values())  # 创建一个无限循环的开局模式迭代器
        #
        # while True:
        #     current_opening = next(opening_cycle)  # 获取下一个开局模式
        #     for location in current_opening:  # 遍历开局模式中的位置
        #         move = self.board.location_to_move(location)  # 将位置转换为棋盘上的移动值
        #         self.board.do_move(move)  # 执行移动
        #     if self.is_shown:
        #         self.board.visual()  # 如果需要展示棋盘，则调用展示方法
        #     break  # 退出循环，开始游戏


        while True:
            current_player = self.board.current_player  # 获取当前玩家
            player_in_turn = players[current_player]  # 当前玩家的信息
            start_time = time.time()
            move = player_in_turn.get_action(self.board)  # 基于MCTS的AI下一步落子
            end_time = time.time()
            move_time = end_time - start_time
            
            if current_player == p1:
                self.player1_times.append(move_time)
            else:
                self.player2_times.append(move_time)
            
            self.board.do_move(move)  # 根据下一步落子的状态更新棋盘各参数
            
            if self.is_shown:
                print(f"落子时间: {move_time:.2f} 秒")
                self.board.visual()
            
            end, winner = self.board.game_end()  # 判断当前棋局是否结束
            if end:
                win = winner
                break

        self.save_average_times()
        
        if self.is_shown:
            if win==self.board.start_player:
                print('胜利者 : 黑方（先手） ')
            else:
                print('胜利者 : 白方（后手）')
            print(f"winner : \t{winner}")
        

        # 保存有效分支因子到文件中
        print(f"玩家1平均有效分支因子: {player1.get_average_branching_factor():.2f}")
        print(f"玩家2平均有效分支因子: {player2.get_average_branching_factor():.2f}")


        with open(f"玩家{i}有效分支因子.txt", "a", encoding="utf-8") as f:
            f.write(f"玩家{i}平均有效分支因子: {player1.get_average_branching_factor():.2f}\n")
        with open(f"玩家{j}有效分支因子.txt", "a", encoding="utf-8") as f:
            f.write(f"玩家{j}平均有效分支因子: {player2.get_average_branching_factor():.2f}\n")

        print("有效分支因子已保存到有效分支因子.txt 文件中")
        return win

    def save_average_times(self):
        avg_time1 = sum(self.player1_times) / len(self.player1_times) if self.player1_times else 0
        avg_time2 = sum(self.player2_times) / len(self.player2_times) if self.player2_times else 0
        
        result = {
            "player1_avg_time": avg_time1,
            "player2_avg_time": avg_time2
        }
        
        # 保存平均落子时间到文件中
        with open(f"玩家{self.player1_id}平均落子时间.txt", "a") as f:
            f.write(f"玩家{self.player1_id}平均落子时间: {avg_time1:.2f} 秒\n")
        with open(f"玩家{self.player2_id}平均落子时间.txt", "a") as f:
            f.write(f"玩家{self.player2_id}平均落子时间: {avg_time2:.2f} 秒\n")
        
        print(f"玩家{self.player1_id}平均落子时间: {avg_time1:.2f} 秒")
        print(f"玩家{self.player2_id}平均落子时间: {avg_time2:.2f} 秒")
        print("平均落子时间已保存到平均落子时间.txt 文件中")
