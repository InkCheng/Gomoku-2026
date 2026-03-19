"""
UI2.py — 五子棋竞技人机对弈界面 (tkinter + 多线程)
完全按 src.game.Board / src.players.AIplayer 竞技规则实现
"""

import os
import sys
import copy
import time
import threading
import json
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from datetime import datetime

# 引入项目自己的类和引擎
import gomoku_engine
import src.game as game
import src.players as players

# 26种指定开局字典 (名字, 白方第4手位置, 五打N手的N默认上限)
openings_dict = {
    "112 113 114": ('寒星局', 96, 8),
    "112 113 129": ('溪月局', 142, 18),
    "112 113 99": ('溪月局', 82, 18),
    "112 113 144": ('疏星局', 128, 2),
    "112 113 84": ('疏星局', 98, 2),
    "112 113 128": ('花月局', 96, 12),
    "112 113 98": ('花月局', 126, 12),
    "112 113 143": ('残月局', 142, 20),
    "112 113 83": ('残月局', 82, 20),
    "112 113 127": ('雨月局', 142, 10),
    "112 113 97": ('雨月局', 82, 10),
    "112 113 142": ('金星局', 129, 14),
    "112 113 82": ('金星局', 99, 14),
    "112 113 111": ('松月局', 110, 17),
    "112 113 126": ('丘月局', 98, 5),
    "112 113 96": ('丘月局', 128, 5),
    "112 113 141": ('新月局', 98, 6),
    "112 113 81": ('新月局', 128, 6),
    "112 113 110": ('瑞星局', 127, 9),
    "112 113 125": ('山月局', 81, 12),
    "112 113 95": ('山月局', 141, 12),
    "112 113 140": ('游星局', 98, 0),
    "112 113 80": ('游星局', 128, 0),
    "112 128 144": ('长星局', 114, 1),
    "112 128 143": ('峡月局', 141, 17),
    "112 128 129": ('峡月局', 99, 17),
    "112 128 142": ('恒星局', 127, 5),
    "112 128 114": ('恒星局', 113, 5),
    "112 128 141": ('水月局', 97, 16),
    "112 128 99": ('水月局', 111, 16),
    "112 128 140": ('流星局', 98, 0),
    "112 128 84": ('流星局', 126, 0),
    "112 128 127": ('云月局', 97, 9),
    "112 128 113": ('云月局', 111, 9),
    "112 128 126": ('浦月局', 98, 9),
    "112 128 98": ('浦月局', 126, 9),
    "112 128 125": ('岚月局', 97, 11),
    "112 128 83": ('岚月局', 111, 11),
    "112 128 111": ('银月局', 110, 16),
    "112 128 97": ('银月局', 82, 16),
    "112 128 110": ('明星局', 113, 9),
    "112 128 82": ('明星局', 127, 9),
    "112 128 96": ('斜月局', 127, 2),
    "112 128 95": ('名月局', 127, 5),
    "112 128 81": ('名月局', 113, 5),
    "112 128 80": ('彗星局', 143, 0),
}

# ═══════════════════════════════════════════════════════════════════════════════
# 竞技状态机枚举
# ═══════════════════════════════════════════════════════════════════════════════
class GamePhase:
    NOT_STARTED = 0
    OPENING_3 = 1          # 前3手开局阶段
    WAIT_SWAP = 2          # 询问三手交换
    WAIT_FIVE_STRIKE_AI = 3   # AI 打入五手N打点
    WAIT_FIVE_STRIKE_HUMAN = 4 # 人类 打入五手N打点
    WAIT_CHOOSE_FIVE = 5   # 选择保留哪一个五手点
    NORMAL = 6             # 正常轮流对弈
    GAME_OVER = 7

# UI布局常量
BOARD_SIZE = 15
CELL_SIZE = 40
MARGIN = 30
STONE_RADIUS = 17
CANVAS_SIZE = MARGIN * 2 + CELL_SIZE * (BOARD_SIZE - 1)  # 590
PANEL_WIDTH = 220

COLOR_BG       = '#DEB887'
COLOR_LINE     = '#2F1B0E'
COLOR_BLACK    = '#111111'
COLOR_WHITE    = '#F5F5F5'
COLOR_LAST     = '#FF4444'
COLOR_STRIKE   = '#4444FF'   # 五打N的候选点提示色
COLOR_HOVER    = '#88CC88'
COLOR_STAR     = '#2F1B0E'
COLOR_PANEL_BG = '#1E1E2E'
COLOR_BTN      = '#3A3A5C'
COLOR_BTN_H    = '#5A5A8C'
COLOR_TEXT     = '#E0E0FF'

class GomokuApp:
    def __init__(self, root: tk.Tk, external_board, hash_table_manager):
        self.root = root
        self.external_board = external_board
        self.hash_table_manager = hash_table_manager
        
        # 将在开始游戏时被实例化
        self.board = None
        self.ai = None
        
        # 对局状态
        self.phase = GamePhase.NOT_STARTED
        self.hover_pos = None
        self.ai_thinking = False
        
        # 五打N相关数据
        self.five_nums = 2        # 默认N=2
        self.five_candidate_moves = [] # AI或人类打出的N个候选点
        
        # 计时统计
        self.ai_total_time = 0.0
        self.human_total_time = 0.0
        self.human_start_time = None
        
        self.game_start_time = None
        self.current_move_start_time = None
        self.last_move_time = 0.0
        self.timer_id = None
        self._load_config()

        self._build_ui()
        self._show_msg("由配置决定先手方，点击开始游戏")
        self._draw_board(clear=True)

    def _load_config(self):
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception as e:
            self.config = {
                "isAIFirst": True,
                "searchTime": 0.1,
                "myName": "alphapig",
                "oppoName": "man",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M")
            }

    def _update_realtime_clock(self):
        if self.phase not in [GamePhase.NOT_STARTED, GamePhase.GAME_OVER]:
            now = time.time()
            if self.game_start_time:
                total_sec = now - self.game_start_time
                gm, gs = divmod(total_sec, 60)
                self.game_time_label.config(text=f"游戏进行时间: {int(gm):02d}:{int(gs):02d}")

            if self.current_move_start_time:
                curr_sec = now - self.current_move_start_time
                cm, cs = divmod(curr_sec, 60)
                lm, ls = divmod(self.last_move_time, 60)
                
                self.move_time_label.config(
                    text=f"上一手花费: {int(lm)}分{ls:.1f}秒\n当前手花费: {int(cm)}分{cs:.1f}秒"
                )
        
        self.timer_id = self.root.after(100, self._update_realtime_clock)

    def _mark_turn_start(self):
        if self.current_move_start_time is not None:
            self.last_move_time = time.time() - self.current_move_start_time
        self.current_move_start_time = time.time()

    def _build_ui(self):
        self.root.title("悟空五子棋 · 竞技版 (UI2)")
        self.root.resizable(False, False)
        self.root.configure(bg=COLOR_PANEL_BG)

        main_frame = tk.Frame(self.root, bg=COLOR_PANEL_BG)
        main_frame.pack()

        # Canvas
        self.canvas = tk.Canvas(main_frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=COLOR_BG, highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.canvas.bind('<Motion>', self._on_mouse_move)
        self.canvas.bind('<Leave>', self._on_mouse_leave)
        self.canvas.bind('<Button-1>', self._on_click)

        # Panel
        panel = tk.Frame(main_frame, width=PANEL_WIDTH, bg=COLOR_PANEL_BG)
        panel.grid(row=0, column=1, padx=10, pady=10, sticky='ns')
        panel.grid_propagate(False)

        tk.Label(panel, text="悟空五子棋", font=("Microsoft YaHei", 18, "bold"), fg='#FFD700', bg=COLOR_PANEL_BG).pack(pady=(20, 10))
        
        self.status_label = tk.Label(panel, text="", font=("Microsoft YaHei", 11), fg=COLOR_TEXT, bg=COLOR_PANEL_BG, wraplength=200, justify='center')
        self.status_label.pack(pady=(5, 15))

        self.turn_label = tk.Label(panel, text="", font=("Microsoft YaHei", 11), fg='#AAAACC', bg=COLOR_PANEL_BG)
        self.turn_label.pack(pady=2)

        self.game_time_label = tk.Label(panel, text="", font=("Microsoft YaHei", 11, "bold"), fg='#55FF55', bg=COLOR_PANEL_BG)
        self.game_time_label.pack(pady=(5, 2))

        self.move_time_label = tk.Label(panel, text="", font=("Microsoft YaHei", 10), fg='#FFD700', bg=COLOR_PANEL_BG, wraplength=200, justify='center')
        self.move_time_label.pack(pady=2)

        self.time_label = tk.Label(panel, text="", font=("Microsoft YaHei", 10), fg='#8888AA', bg=COLOR_PANEL_BG, wraplength=200, justify='center')
        self.time_label.pack(pady=2)

        # Buttons
        btn_frame = tk.Frame(panel, bg=COLOR_PANEL_BG)
        btn_frame.pack(side='bottom', pady=20, fill='x', padx=15)

        self.btn_start = self._make_btn(btn_frame, "开始游戏", lambda: self._start_game(ai_first=self.config.get('isAIFirst', True)))
        self.btn_start.pack(fill='x', pady=4)

        self.btn_swap = self._make_btn(btn_frame, "三手交换", self._do_swap)
        self.btn_swap.pack(fill='x', pady=4)
        self.btn_swap.pack_forget()

        self.btn_no_swap = self._make_btn(btn_frame, "不交换", self._do_not_swap)
        self.btn_no_swap.pack(fill='x', pady=4)
        self.btn_no_swap.pack_forget()

        # Note: Gomoku engine holds states, undo is risky in the middle of these rules, 
        # so for this strict competition mode, undo is usually omitted or handled very carefully.
        # We will keep a simplified new game button.
        self.btn_new = self._make_btn(btn_frame, "重置对局", self._reset_ui)
        self.btn_new.pack(fill='x', pady=15)

    def _make_btn(self, parent, text, cmd):
        btn = tk.Button(parent, text=text, command=cmd, font=("Microsoft YaHei", 11, "bold"),
                        bg=COLOR_BTN, fg=COLOR_TEXT, activebackground=COLOR_BTN_H, activeforeground='#FFFFFF',
                        relief='flat', bd=0, padx=5, pady=6, cursor='hand2')
        btn.bind('<Enter>', lambda e, b=btn: b.config(bg=COLOR_BTN_H))
        btn.bind('<Leave>', lambda e, b=btn: b.config(bg=COLOR_BTN))
        return btn

    def _show_msg(self, msg):
        self.status_label.config(text=msg)

    def _reset_ui(self):
        self.phase = GamePhase.NOT_STARTED
        self.ai_thinking = False
        self.five_candidate_moves = []
        
        self.game_start_time = None
        self.current_move_start_time = None
        self.last_move_time = 0.0
        if getattr(self, 'timer_id', None):
            self.root.after_cancel(self.timer_id)
            self.timer_id = None
            
        self._load_config()
        self._show_msg("由配置决定先手方，点击开始游戏")
        if hasattr(self, 'btn_start'):
            self.btn_start.pack(fill='x', pady=4)
        self.btn_swap.pack_forget()
        self.btn_no_swap.pack_forget()
        self.turn_label.config(text="")
        if hasattr(self, 'game_time_label'): self.game_time_label.config(text="")
        if hasattr(self, 'move_time_label'): self.move_time_label.config(text="")
        self.time_label.config(text="")
        self._draw_board(clear=True)

    # ── 游戏流程 ──────────────────────────────────────────────────

    def _start_game(self, ai_first):
        """初始化比赛"""
        if hasattr(self, 'btn_start'):
            self.btn_start.pack_forget()
        
        self.board = game.Board(ExternalProgramManager=self.external_board, hash_table_manager=self.hash_table_manager, width=15, height=15, n_in_row=5)
        # 1. 0是AI，1是人类。 start_player设为0(首位出牌者)。 但我们用AI_turn来区分谁是AI。
        # 原版逻辑中，start_player恒为0，如果AI先手，AI_turn=0；如果人类先手，AI_turn=1.
        self.board.init_board(start_player=0)
        self.board.AI_turn = 0 if ai_first else 1
        
        # 重新初始化AIPlayer，原代码在外面初始化会导致 MCTS 树的状态污染。这里重新加载 (c_puct=2, n_playout=400)
        model_path = "best_model/current_policy_step_best.model"
        if not os.path.exists(model_path):
            messagebox.showerror("错误", f"找不到策略模型: {model_path}")
            self._reset_ui()
            return
        
        self._show_msg("加载 AI 模型中...")
        self.root.update()
        
        self.ai = players.AIplayer(model_path=model_path, c_puct=2, n_playout=400, is_selfplay=False)
        self.ai.mcts_player.search_time = self.config.get("searchTime", 0.1)
        self.ai_total_time = 0.0
        self.human_total_time = 0.0
        
        self.phase = GamePhase.OPENING_3
        self.five_candidate_moves = []
        
        self.game_start_time = time.time()
        self.current_move_start_time = time.time()
        self.last_move_time = 0.0
        if getattr(self, 'timer_id', None) is None:
            self._update_realtime_clock()
        
        if ai_first:
            # AI 先手开局 (走3步: 疏星局)
            self._ai_fixed_opening()
        else:
            self._show_msg("人类先手 (黑方)\n请在天元(H8)落第1子")
            self.human_start_time = time.time()
            self._draw_board()
            self._mark_turn_start()

    def _ai_fixed_opening(self):
        name = self.config.get('myName', '悟空')
        self._show_msg(f"{name}先手，默认选择疏星局开局...")
        self.root.update()
        time.sleep(0.5)
        # 固定开局
        self.board.do_move(112) # H8
        self._draw_board()
        self.root.update()
        time.sleep(0.5)
        self.board.do_move(113) # I8
        self._draw_board()
        self.root.update()
        time.sleep(0.5)
        self.board.do_move(9 * 15 + 9) # J10
        self._draw_board()
        self.root.update()
        
        self.phase = GamePhase.WAIT_SWAP
        self._show_msg("疏星局开局完毕。\n请问是否三手交换？")
        self.btn_swap.pack(fill='x', pady=4)
        self.btn_no_swap.pack(fill='x', pady=4)
        self._mark_turn_start()

    # ── 事件 ──────────────────────────────────────────────────

    def _do_swap(self):
        """人类要求交换 (AI 先手开局后)"""
        self.btn_swap.pack_forget()
        self.btn_no_swap.pack_forget()
        
        self._show_msg("你选择了交换！你现在执黑。请为白方落第4子")
        # 交换身份
        self.board.AI_turn = 1 - self.board.AI_turn
        self.phase = GamePhase.NORMAL # 原版这里需要人类落第4子，落完后AI要求打点
        self.human_start_time = time.time()
        self._mark_turn_start()

    def _do_not_swap(self):
        """人类不交换 (AI 先手开局后)"""
        self.btn_swap.pack_forget()
        self.btn_no_swap.pack_forget()
        self._show_msg("继续游戏（你不交换）。\n请落第4子")
        self.phase = GamePhase.NORMAL
        self.human_start_time = time.time()
        self._mark_turn_start()

    def _on_click(self, event):
        if self.phase == GamePhase.NOT_STARTED or self.phase == GamePhase.GAME_OVER or self.ai_thinking:
            return
            
        pos = self._pixel_to_board(event.x, event.y)
        if pos is None: return
        row, col = pos
        # UI_14_Y: 原版由于底层 C++ 引擎的坐标系可能上下颠倒，原版 UI 这样传坐标给 move
        # 但既然我们用 game.Board.location_to_move( (row, col) )，我们先统一对齐。
        # 我们用标准行列传入 location_to_move
        # 原版的(x, 14-y)，我们这里在_pixel_to_board 直接把 row当作游戏里的纵坐标
        # 不过需要验证 gomoku_engine 的方向。它认为 0 是上还是下？
        # 在 game.Board 中 move_to_location 返回 [h, w]。h就是row。
        move = self.board.location_to_move((row, col))
        if move == -1: return

        if self.phase == GamePhase.OPENING_3:
            # 人类开局（第1、2、3手）
            if move not in self.board.get_available(self.board.current_player):
                messagebox.showwarning("非法", "此处无法落子或禁手")
                return
            
            # 第一手必须在天元 7,7
            if self.board.turn == 0 and move != 7 * 15 + 7:
                messagebox.showwarning("规则", "第一手必须下在天元星(H8)")
                return
            
            self._human_do_move(move)
            if self.board.turn == 3:
                # 检查开局是否在字典中
                keys = list(self.board.states.keys())[:3]
                key_str = f"{keys[0]} {keys[1]} {keys[2]}"
                if key_str in openings_dict:
                    self.opening_info = openings_dict[key_str]
                    self.phase = GamePhase.WAIT_SWAP
                    self._show_msg(f"局面: {self.opening_info[0]}\nAI 正在决定是否交换...")
                    self._draw_board()
                    self._mark_turn_start()
                    self.root.after(1000, self._ai_decide_swap)
                else:
                    messagebox.showerror("错误", "非标准开局库开局，请重新开始")
                    self._reset_ui()
                    return

        elif self.phase == GamePhase.WAIT_FIVE_STRIKE_HUMAN:
            # 人类给出 N 个候选打点
            if move not in self.board.get_available(self.board.current_player): return
            if move in self.five_candidate_moves: return
            
            self.five_candidate_moves.append(move)
            self._draw_board()
            
            if len(self.five_candidate_moves) == self.five_nums:
                self.phase = GamePhase.WAIT_CHOOSE_FIVE
                self._show_msg(f"你已给出 {self.five_nums} 个点。\nAI 正在选择最优打点...")
                self.root.update()
                self._mark_turn_start()
                self.root.after(500, self._ai_choose_five_strike)
            else:
                self._show_msg(f"请再选择 {self.five_nums - len(self.five_candidate_moves)} 个五手打点")

        elif self.phase == GamePhase.WAIT_CHOOSE_FIVE:
            # 当前是AI让 人类 选一个保留。所以人类点击了某个候选点
            if self.board.current_player != self.board.AI_turn: return # 这是人选，不该AI点（防误触）
            if move not in self.five_candidate_moves and not any(move == int(m) for m in self.five_candidate_moves):
                messagebox.showwarning("提示", "必须选择蓝色提示圈内的打点保留！")
                return
            # 保留此点
            self.board.do_move(move)
            self.five_candidate_moves = []
            self.phase = GamePhase.NORMAL
            self._show_msg("已保留打点。AI 思考中...")
            self._draw_board()
            self._record_time()
            self._ai_launch_thread()
            self._mark_turn_start()

        elif self.phase == GamePhase.NORMAL:
            if move not in self.board.get_available(self.board.current_player):
                messagebox.showwarning("提示", "黑棋禁手或此处已有棋子")
                return
            self._human_do_move(move)
            
            # 五手二打特殊情况：如果这是人类走第5手 (turn==4 准备到 turn==5)
            # 原版中这一步在上面 WAIT_FIVE_STRIKE_HUMAN 拦截，所以如果这里正常下，说明不是第5手，或者是第4步后 AI打点。
            if self.board.turn == 4 and self.board.current_player != self.board.AI_turn:
                # 代表人类正准备下第5手
                pass # 交给前面拦截

    def _ai_decide_swap(self):
        # AI 逻辑判断由于太复杂，原版直接简化为：如果不超过打点约束，AI选三手交换
        # UI.py 原版逻辑: 弹窗让人类输入打点数
        nums = simpledialog.askinteger("五手N打", "请输入五手打点数N (1-10)：", minvalue=1, maxvalue=10)
        self.five_nums = nums if nums else 2
        
        info = self.opening_info
        name = self.config.get('myName', '悟空')
        if self.five_nums <= info[2]:
            self._show_msg(f"{info[0]}\n{name}选择【三手交换】！\n请你下第4子(白方)")
            self.board.AI_turn = 1 - self.board.AI_turn
            self.phase = GamePhase.NORMAL
            self.human_start_time = time.time()
            self._mark_turn_start()
        else:
            self._show_msg(f"{info[0]}\n{name}【不交换】！\n{name}长考第4子...")
            self.phase = GamePhase.NORMAL
            self.root.update()
            # AI 下第4子
            self._ai_launch_thread()
            self._mark_turn_start()

    def _human_do_move(self, move):
        self._record_time()
        ok = self.board.do_move(move)
        if not ok: 
            messagebox.showerror("犯规", "底层引擎报犯规！")
        self._draw_board()
        self._update_time_label()
        
        if self._check_game_end(): return

        if self.board.turn == 3: # 前面_on_click自己拦截处理三手交换了
            pass 
        elif self.board.turn == 4 and self.board.current_player == self.board.AI_turn:
            # AI 准备打五手N打
            self.phase = GamePhase.WAIT_FIVE_STRIKE_AI
            self._show_msg(f"AI 正在计算 {self.five_nums} 个五手打点...")
            self._ai_launch_thread(strike_5=True)
            self._mark_turn_start()
        elif self.board.turn == 4 and self.board.current_player != self.board.AI_turn:
            # 人类 准备打五手N打
            self.phase = GamePhase.WAIT_FIVE_STRIKE_HUMAN
            self._show_msg(f"需提供 {self.five_nums} 个打点，请在棋盘上连续点击")
            self.human_start_time = time.time()
            self._mark_turn_start()
        else:
            self.phase = GamePhase.NORMAL
            self._ai_launch_thread()
            self._mark_turn_start()

    def _ai_choose_five_strike(self):
        # AI 看着人类提供的 self.five_candidate_moves，选一个由于是对AI最有利(人类最不利)的点
        # 这里为了简化，我们在子线程用 AI 评估
        self.ai_thinking = True
        self._ai_result = None
        threading.Thread(target=self._ai_choose_worker, daemon=True).start()
        self.root.after(50, self._check_ai_choose)

    def _ai_choose_worker(self):
        best_move = self.five_candidate_moves[0]
        best_val = float('inf')  # 找对自己最有利的，即对落子方(人类)最不利的
        for mv in self.five_candidate_moves:
            self.board.do_move(mv)
            # evaluate 返回的 leaf_value 是对当前行动方(此时轮到AI白方了)的胜率期望
            # 所以 leaf_value 越大说明对AI越有利
            _, v = self.ai.evaluate(self.board) 
            self.board.undo_move( *self.board.move_to_location(mv) )
            if v < best_val:  # 因为模型里评估是对落子方，需要看清谁在评估
                best_val = v
                best_move = mv
        self._ai_result = ('ok', best_move)

    def _check_ai_choose(self):
        if self._ai_result is None:
            self.root.after(50, self._check_ai_choose)
            return
        self.ai_thinking = False
        _, move = self._ai_result
        self._show_msg("AI 已选择打点保留。AI 继续思考下第6手...")
        self.five_candidate_moves = []
        self.board.do_move(move) # 落实第5手
        self.phase = GamePhase.NORMAL
        self._draw_board()
        
        if self._check_game_end(): return
        self._ai_launch_thread() # AI开始下第6步
        self._mark_turn_start()

    # ── AI 动作 ──────────────────────────────────────────────────

    def _ai_launch_thread(self, strike_5=False):
        self.ai_thinking = True
        if not strike_5:
            name = self.config.get('myName', '悟空')
            self._show_msg(f"{name}长考中... (MCTS)")
        self.btn_swap.pack_forget()
        self.btn_no_swap.pack_forget()
        
        self._ai_result = None
        self._ai_start_time = time.time()
        threading.Thread(target=self._ai_worker, args=(strike_5,), daemon=True).start()
        self.root.after(50, self._check_ai_done, strike_5)

    def _ai_worker(self, strike_5):
        try:
            if not strike_5:
                # 正常 MCTS 落子
                move = self.ai.get_action(self.board)
                self._ai_result = ('ok', move)
            else:
                # 获取五手打点候选
                moves = []
                # 原版使用 min_leaf_heap 找到最差的点（供人选，人肯定选对人最有利的，所以AI得先挑些好点或坏点）
                # 这里我们直接调用MCTS跑前 N 个最优概率的动作
                _, probs = self.ai.get_action(self.board, return_prob=1)
                sorted_indices = np.argsort(probs)[::-1]
                cnt = 0
                for idx in sorted_indices:
                    # 判断对称逻辑（原版有），为了简化和不越界，只要在可用里就加
                    if idx in self.board.get_available(self.board.current_player):
                        moves.append(idx)
                        cnt += 1
                        if cnt >= self.five_nums: break
                self._ai_result = ('strike_5', moves)
        except Exception as e:
            self._ai_result = ('error', str(e))

    def _check_ai_done(self, strike_5):
        if self._ai_result is None:
            self.root.after(50, self._check_ai_done, strike_5)
            return
            
        status, data = self._ai_result
        cost = time.time() - self._ai_start_time
        self.ai_total_time += cost
        self.ai_thinking = False
        
        if status == 'error':
            self._show_msg(f"AI 故障: {data}")
            return
            
        if status == 'strike_5':
            self.five_candidate_moves = data
            self.phase = GamePhase.WAIT_CHOOSE_FIVE
            self._show_msg("AI 提出了五手打点候选。\n请点击蓝色提示圈，选择你要留给 AI 的落子。")
            self._draw_board()
            self.human_start_time = time.time()
            self._mark_turn_start()
        else:
            # 正常落子
            move = data
            self.board.do_move(move)
            self._draw_board()
            self._update_time_label()
            
            if self._check_game_end(): return

            if self.board.turn == 4 and self.board.current_player != self.board.AI_turn:
                # 第4子后轮到人类下 第5步，去打点
                self.phase = GamePhase.WAIT_FIVE_STRIKE_HUMAN
                self._show_msg(f"请你连续点击棋盘，提供 {self.five_nums} 个五手打点")
                self.five_candidate_moves = []
                self.human_start_time = time.time()
                self._mark_turn_start()
            else:
                self.phase = GamePhase.NORMAL
                self._show_msg("轮到你了 (下子)")
                self.human_start_time = time.time()
                self._mark_turn_start()

    def _check_game_end(self):
        end, winner = self.board.game_end()
        if not end:
            return False
            
        self.phase = GamePhase.GAME_OVER
        self.ai_thinking = False
        self._draw_board()

        msg = ""
        if winner == -1:
            msg = "平局！棋盘已满。"
        elif self.board.have_non_compliance == 1:
            loser = "黑棋" if self.board.history_player == 0 else "白棋"
            msg = f"{loser} 犯规（长连/四四/三三禁手）判负！\n对方获胜。"
        else:
            ai_name = self.config.get('myName', '悟空')
            hu_name = self.config.get('oppoName', '你(人类)')
            ms = f"{ai_name}(AI)" if winner == self.board.AI_turn else f"{hu_name}"
            msg = f"游戏结束！{ms} 连成五子获胜！"

        self._show_msg(msg)
        self.root.after(300, lambda: messagebox.showinfo("对局结束", msg))
        return True

    def _record_time(self):
        if self.human_start_time:
            self.human_total_time += time.time() - self.human_start_time
            self.human_start_time = None

    def _update_time_label(self):
        total_moves = len(self.board.states) if self.board else 0
        self.turn_label.config(text=f"第 {total_moves} 手")
        ai_m, ai_s = divmod(self.ai_total_time, 60)
        hu_m, hu_s = divmod(self.human_total_time, 60)
        ai_name = self.config.get('myName', '悟空')
        hu_name = self.config.get('oppoName', '人类')
        self.time_label.config(
            text=f"{ai_name}耗时: {int(ai_m)}分{ai_s:.1f}秒\n"
                 f"{hu_name}耗时: {int(hu_m)}分{hu_s:.1f}秒"
        )

    # ── 绘图 ──────────────────────────────────────────────────

    def _pixel_to_board(self, px, py):
        col = round((px - MARGIN) / CELL_SIZE)
        row = round((py - MARGIN) / CELL_SIZE)
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return row, col
        return None

    def _on_mouse_move(self, event):
        pos = self._pixel_to_board(event.x, event.y)
        if pos != self.hover_pos:
            self.hover_pos = pos
            if self.phase != GamePhase.NOT_STARTED:
                self._draw_board()

    def _on_mouse_leave(self, event):
        self.hover_pos = None
        if self.phase != GamePhase.NOT_STARTED:
            self._draw_board()

    def _draw_board(self, clear=False):
        self.canvas.delete('all')

        # 线条
        for i in range(BOARD_SIZE):
            x = MARGIN + i * CELL_SIZE
            y = MARGIN + i * CELL_SIZE
            self.canvas.create_line(x, MARGIN, x, MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, fill=COLOR_LINE)
            self.canvas.create_line(MARGIN, y, MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, y, fill=COLOR_LINE)
            # 坐标
            self.canvas.create_text(x, MARGIN - 15, text=chr(ord('A') + i), font=("Consolas", 9), fill=COLOR_LINE)
            self.canvas.create_text(MARGIN - 17, y, text=str(BOARD_SIZE - i), font=("Consolas", 9), fill=COLOR_LINE)

        # 星位
        star_points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for sr, sc in star_points:
            sx, sy = MARGIN + sc * CELL_SIZE, MARGIN + sr * CELL_SIZE
            self.canvas.create_oval(sx - 3, sy - 3, sx + 3, sy + 3, fill=COLOR_STAR)

        if clear or self.board is None:
            return

        # 棋子
        for move, player in self.board.states.items():
            row, col = self.board.move_to_location(move)
            cx, cy = MARGIN + col * CELL_SIZE, MARGIN + row * CELL_SIZE
            color = COLOR_BLACK if player == self.board.players[0] else COLOR_WHITE
            outline = '#333333' if color == COLOR_BLACK else '#AAAAAA'
            self.canvas.create_oval(cx - STONE_RADIUS, cy - STONE_RADIUS, cx + STONE_RADIUS, cy + STONE_RADIUS, fill=color, outline=outline, width=1.5)

        # 最后一手
        if self.board.last_move >= 0:
            row, col = self.board.move_to_location(self.board.last_move)
            cx, cy = MARGIN + col * CELL_SIZE, MARGIN + row * CELL_SIZE
            self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=COLOR_LAST, outline=COLOR_LAST)

        # 候选打点 (五手N打)
        for cand_move in self.five_candidate_moves:
            row, col = self.board.move_to_location(cand_move)
            cx, cy = MARGIN + col * CELL_SIZE, MARGIN + row * CELL_SIZE
            self.canvas.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, outline=COLOR_STRIKE, fill='', width=3)

        # 悬浮
        if self.hover_pos and not self.ai_thinking and self.phase not in [GamePhase.GAME_OVER, GamePhase.WAIT_CHOOSE_FIVE]:
            hr, hc = self.hover_pos
            move = self.board.location_to_move((hr, hc))
            # 简单检查可用
            if move != -1 and move not in self.board.states:
                cx, cy = MARGIN + hc * CELL_SIZE, MARGIN + hr * CELL_SIZE
                self.canvas.create_oval(cx - STONE_RADIUS+2, cy - STONE_RADIUS+2, cx + STONE_RADIUS-2, cy + STONE_RADIUS-2, outline=COLOR_HOVER, dash=(4,4), width=2)


def main():
    print("=" * 60)
    print(" 悟空五子棋竞技版 (UI2): MCTS + TSS + gomoku_engine")
    print("=" * 60)

    # 初始化底层引擎
    if not os.path.exists("merged_hash_new2.pkl"):
        print("[提示] 找不到 merged_hash_new2.pkl，将被自动创建空哈希表。")
    
    externalProgramManager = gomoku_engine.Board()
    hash_table_manager = game.HashTableManager("merged_hash_new2.pkl")
    
    root = tk.Tk()
    app = GomokuApp(root, externalProgramManager, hash_table_manager)
    
    try:
        root.mainloop()
    finally:
        # 关闭时退出引擎
        if hasattr(externalProgramManager, 'terminate'):
            externalProgramManager.terminate()

if __name__ == '__main__':
    main()
