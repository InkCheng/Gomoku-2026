"""
UI2_pyqt.py — 五子棋竞技人机对弈界面 (PyQt5 + 多线程并发)

完全按 src.game.Board / src.players.AIPlayerParallel 竞技规则实现。
并发设计:
  1. UI / AI 隔离 — AIWorkerThread (QThread)
  2. MCTS 搜索并行化 — MCTSParallel (threading.Thread x N)
  3. NN 推理批处理 — NNBatchServer (daemon thread)
"""

import os
import sys
import copy
import time
import json
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QMessageBox, QInputDialog, QFrame, QSizePolicy
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, pyqtSlot, QPoint, QRectF
)
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QFontMetrics, QCursor
)
from datetime import datetime

# 引入项目类
import gomoku_engine
import src.game as game
import src.players as players

# ═══════════════════════════════════════════════════════════════
# 26 种指定开局字典
# ═══════════════════════════════════════════════════════════════
openings_dict = {
    "112 113 114": ('寒星局', 96, 8),
    "112 113 129": ('溪月局', 142, 18),
    "112 113 99":  ('溪月局', 82, 18),
    "112 113 144": ('疏星局', 128, 2),
    "112 113 84":  ('疏星局', 98, 2),
    "112 113 128": ('花月局', 96, 12),
    "112 113 98":  ('花月局', 126, 12),
    "112 113 143": ('残月局', 142, 20),
    "112 113 83":  ('残月局', 82, 20),
    "112 113 127": ('雨月局', 142, 10),
    "112 113 97":  ('雨月局', 82, 10),
    "112 113 142": ('金星局', 129, 14),
    "112 113 82":  ('金星局', 99, 14),
    "112 113 111": ('松月局', 110, 17),
    "112 113 126": ('丘月局', 98, 5),
    "112 113 96":  ('丘月局', 128, 5),
    "112 113 141": ('新月局', 98, 6),
    "112 113 81":  ('新月局', 128, 6),
    "112 113 110": ('瑞星局', 127, 9),
    "112 113 125": ('山月局', 81, 12),
    "112 113 95":  ('山月局', 141, 12),
    "112 113 140": ('游星局', 98, 0),
    "112 113 80":  ('游星局', 128, 0),
    "112 128 144": ('长星局', 114, 1),
    "112 128 143": ('峡月局', 141, 17),
    "112 128 129": ('峡月局', 99, 17),
    "112 128 142": ('恒星局', 127, 5),
    "112 128 114": ('恒星局', 113, 5),
    "112 128 141": ('水月局', 97, 16),
    "112 128 99":  ('水月局', 111, 16),
    "112 128 140": ('流星局', 98, 0),
    "112 128 84":  ('流星局', 126, 0),
    "112 128 127": ('云月局', 97, 9),
    "112 128 113": ('云月局', 111, 9),
    "112 128 126": ('浦月局', 98, 9),
    "112 128 98":  ('浦月局', 126, 9),
    "112 128 125": ('岚月局', 97, 11),
    "112 128 83":  ('岚月局', 111, 11),
    "112 128 111": ('银月局', 110, 16),
    "112 128 97":  ('银月局', 82, 16),
    "112 128 110": ('明星局', 113, 9),
    "112 128 82":  ('明星局', 127, 9),
    "112 128 96":  ('斜月局', 127, 2),
    "112 128 95":  ('名月局', 127, 5),
    "112 128 81":  ('名月局', 113, 5),
    "112 128 80":  ('彗星局', 143, 0),
}

# ═══════════════════════════════════════════════════════════════
# 竞技状态机枚举
# ═══════════════════════════════════════════════════════════════
class GamePhase:
    NOT_STARTED = 0
    OPENING_3 = 1
    WAIT_SWAP = 2
    WAIT_FIVE_STRIKE_AI = 3
    WAIT_FIVE_STRIKE_HUMAN = 4
    WAIT_CHOOSE_FIVE = 5
    NORMAL = 6
    GAME_OVER = 7

# ═══════════════════════════════════════════════════════════════
# UI 常量
# ═══════════════════════════════════════════════════════════════
BOARD_SIZE = 15
CELL_SIZE = 40
MARGIN = 30
STONE_RADIUS = 17
CANVAS_SIZE = MARGIN * 2 + CELL_SIZE * (BOARD_SIZE - 1)  # 590
PANEL_WIDTH = 240

# 颜色
C_BG       = QColor('#DEB887')
C_LINE     = QColor('#2F1B0E')
C_BLACK    = QColor('#111111')
C_WHITE    = QColor('#F5F5F5')
C_LAST     = QColor('#FF4444')
C_STRIKE   = QColor('#4444FF')
C_HOVER    = QColor('#88CC88')
C_STAR     = QColor('#2F1B0E')
C_PANEL    = QColor('#1E1E2E')
C_BTN      = QColor('#3A3A5C')
C_BTN_H    = QColor('#5A5A8C')
C_TEXT     = QColor('#E0E0FF')


# ═══════════════════════════════════════════════════════════════
# 棋盘绘制 Widget
# ═══════════════════════════════════════════════════════════════
class BoardWidget(QWidget):
    """自定义 QWidget, 用 QPainter 绘制棋盘。"""

    clicked = pyqtSignal(int, int)  # row, col

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(CANVAS_SIZE, CANVAS_SIZE)
        self.setMouseTracking(True)

        self.board = None
        self.hover_pos = None
        self.five_candidates = []
        self.allow_hover = True

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # 背景
        p.fillRect(self.rect(), C_BG)

        # 线条
        pen = QPen(C_LINE, 1)
        p.setPen(pen)
        for i in range(BOARD_SIZE):
            x = MARGIN + i * CELL_SIZE
            y = MARGIN + i * CELL_SIZE
            p.drawLine(x, MARGIN, x, MARGIN + (BOARD_SIZE - 1) * CELL_SIZE)
            p.drawLine(MARGIN, y, MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, y)

        # 坐标标签
        font_coord = QFont("Consolas", 9)
        p.setFont(font_coord)
        p.setPen(C_LINE)
        for i in range(BOARD_SIZE):
            x = MARGIN + i * CELL_SIZE
            y = MARGIN + i * CELL_SIZE
            p.drawText(x - 4, MARGIN - 15, chr(ord('A') + i))
            p.drawText(MARGIN - 22, y + 4, str(BOARD_SIZE - i))

        # 星位
        for sr, sc in [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]:
            sx = MARGIN + sc * CELL_SIZE
            sy = MARGIN + sr * CELL_SIZE
            p.setBrush(QBrush(C_STAR))
            p.setPen(Qt.NoPen)
            p.drawEllipse(sx - 3, sy - 3, 6, 6)

        if self.board is None:
            p.end()
            return

        # 棋子
        for move, player in self.board.states.items():
            row, col = self.board.move_to_location(move)
            cx = MARGIN + col * CELL_SIZE
            cy = MARGIN + row * CELL_SIZE
            if player == self.board.players[0]:
                color = C_BLACK
                outline = QColor('#333333')
            else:
                color = C_WHITE
                outline = QColor('#AAAAAA')
            p.setBrush(QBrush(color))
            p.setPen(QPen(outline, 1.5))
            p.drawEllipse(cx - STONE_RADIUS, cy - STONE_RADIUS,
                          STONE_RADIUS * 2, STONE_RADIUS * 2)

        # 最后一手标记
        if self.board.last_move >= 0:
            row, col = self.board.move_to_location(self.board.last_move)
            cx = MARGIN + col * CELL_SIZE
            cy = MARGIN + row * CELL_SIZE
            p.setBrush(QBrush(C_LAST))
            p.setPen(Qt.NoPen)
            p.drawEllipse(cx - 4, cy - 4, 8, 8)

        # 候选打点
        for cand_move in self.five_candidates:
            row, col = self.board.move_to_location(cand_move)
            cx = MARGIN + col * CELL_SIZE
            cy = MARGIN + row * CELL_SIZE
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(C_STRIKE, 3))
            p.drawEllipse(cx - 7, cy - 7, 14, 14)

        # 悬浮提示
        if self.hover_pos and self.allow_hover:
            hr, hc = self.hover_pos
            move = self.board.location_to_move((hr, hc))
            if move != -1 and move not in self.board.states:
                cx = MARGIN + hc * CELL_SIZE
                cy = MARGIN + hr * CELL_SIZE
                pen_h = QPen(C_HOVER, 2, Qt.DashLine)
                p.setPen(pen_h)
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(cx - STONE_RADIUS + 2, cy - STONE_RADIUS + 2,
                              (STONE_RADIUS - 2) * 2, (STONE_RADIUS - 2) * 2)

        p.end()

    def mouseMoveEvent(self, event):
        pos = self._pixel_to_board(event.x(), event.y())
        if pos != self.hover_pos:
            self.hover_pos = pos
            self.update()

    def leaveEvent(self, event):
        self.hover_pos = None
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self._pixel_to_board(event.x(), event.y())
            if pos:
                self.clicked.emit(pos[0], pos[1])

    def _pixel_to_board(self, px, py):
        col = round((px - MARGIN) / CELL_SIZE)
        row = round((py - MARGIN) / CELL_SIZE)
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return (row, col)
        return None


# ═══════════════════════════════════════════════════════════════
# AI Worker QThread — UI / AI 隔离
# ═══════════════════════════════════════════════════════════════
class AIWorkerThread(QThread):
    """
    在子线程中运行 AI 计算, 完成后通过信号通知主线程。
    防止 MCTS 等重计算阻塞 Qt 事件循环。
    """

    # 信号: (status: str, data: object)
    finished = pyqtSignal(str, object)

    def __init__(self, ai, board, strike_5=False, parent=None):
        super().__init__(parent)
        self.ai = ai
        self.board = board
        self.strike_5 = strike_5

    def run(self):
        try:
            if not self.strike_5:
                move = self.ai.get_action(self.board)
                self.finished.emit('ok', move)
            else:
                _, probs = self.ai.get_action(self.board, return_prob=1)
                sorted_indices = np.argsort(probs)[::-1]
                moves = []
                cnt = 0
                for idx in sorted_indices:
                    if idx in self.board.get_available(
                            self.board.current_player):
                        moves.append(int(idx))
                        cnt += 1
                        if cnt >= self._five_nums:
                            break
                self.finished.emit('strike_5', moves)
        except Exception as e:
            self.finished.emit('error', str(e))


class AIChooseWorkerThread(QThread):
    """
    在子线程中让 AI 评估五手打点候选, 选最优。
    """
    finished = pyqtSignal(int)  # best_move

    def __init__(self, ai, board, candidates, parent=None):
        super().__init__(parent)
        self.ai = ai
        self.board = board
        self.candidates = candidates

    def run(self):
        best_move = self.candidates[0]
        best_val = float('inf')
        for mv in self.candidates:
            self.board.do_move(mv)
            _, v = self.ai.evaluate(self.board)
            self.board.undo_move(*self.board.move_to_location(mv))
            if v < best_val:
                best_val = v
                best_move = mv
        self.finished.emit(best_move)


# ═══════════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════════
class GomokuMainWindow(QMainWindow):

    def __init__(self, external_board, hash_table_manager):
        super().__init__()
        self.external_board = external_board
        self.hash_table_manager = hash_table_manager

        self.board = None
        self.ai = None
        self.phase = GamePhase.NOT_STARTED
        self.ai_thinking = False

        self.five_nums = 2
        self.five_candidate_moves = []

        self.ai_total_time = 0.0
        self.human_total_time = 0.0
        self.human_start_time = None

        self.game_start_time = None
        self.current_move_start_time = None
        self.last_move_time = 0.0

        self._ai_worker = None
        self._ai_choose_worker = None
        self._ai_start_time = 0.0

        self._load_config()
        self._build_ui()
        self._show_msg("由配置决定先手方，点击开始游戏")

        # 实时计时器 (QTimer)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_realtime_clock)
        self._timer.setInterval(100)

    # ── 加载配置 ──
    def _load_config(self):
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception:
            self.config = {
                "isAIFirst": True,
                "searchTime": 0.1,
                "myName": "alphapig",
                "oppoName": "man",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M"),
                "numWorkers": 4,
                "maxBatchSize": 8,
            }

    # ── 构建 UI ──
    def _build_ui(self):
        self.setWindowTitle("悟空五子棋 · 竞技版 (PyQt5 并发)")
        self.setFixedSize(CANVAS_SIZE + PANEL_WIDTH + 30, CANVAS_SIZE + 20)

        central = QWidget()
        central.setStyleSheet(f"background-color: {C_PANEL.name()};")
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        # 棋盘
        self.board_widget = BoardWidget()
        self.board_widget.clicked.connect(self._on_click)
        layout.addWidget(self.board_widget)

        # 右侧面板
        panel = QFrame()
        panel.setFixedWidth(PANEL_WIDTH)
        panel.setStyleSheet(f"background-color: {C_PANEL.name()};")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 20, 10, 20)

        # 标题
        title = QLabel("悟空五子棋")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title.setStyleSheet("color: #FFD700;")
        title.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(title)

        # 状态消息
        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Microsoft YaHei", 11))
        self.status_label.setStyleSheet(f"color: {C_TEXT.name()};")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(self.status_label)

        # 回合
        self.turn_label = QLabel("")
        self.turn_label.setFont(QFont("Microsoft YaHei", 11))
        self.turn_label.setStyleSheet("color: #AAAACC;")
        self.turn_label.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(self.turn_label)

        # 游戏时间
        self.game_time_label = QLabel("")
        self.game_time_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.game_time_label.setStyleSheet("color: #55FF55;")
        self.game_time_label.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(self.game_time_label)

        # 单手耗时
        self.move_time_label = QLabel("")
        self.move_time_label.setFont(QFont("Microsoft YaHei", 10))
        self.move_time_label.setStyleSheet("color: #FFD700;")
        self.move_time_label.setWordWrap(True)
        self.move_time_label.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(self.move_time_label)

        # 总耗时
        self.time_label = QLabel("")
        self.time_label.setFont(QFont("Microsoft YaHei", 10))
        self.time_label.setStyleSheet("color: #8888AA;")
        self.time_label.setWordWrap(True)
        self.time_label.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(self.time_label)

        panel_layout.addStretch()

        # 按钮区
        btn_style = (
            f"QPushButton {{"
            f"  background-color: {C_BTN.name()}; color: {C_TEXT.name()};"
            f"  border: none; padding: 8px; font-size: 14px;"
            f"  font-weight: bold; font-family: 'Microsoft YaHei';"
            f"}}"
            f"QPushButton:hover {{"
            f"  background-color: {C_BTN_H.name()}; color: white;"
            f"}}"
        )

        self.btn_start = QPushButton("开始游戏")
        self.btn_start.setStyleSheet(btn_style)
        self.btn_start.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_start.clicked.connect(
            lambda: self._start_game(
                ai_first=self.config.get('isAIFirst', True)
            )
        )
        panel_layout.addWidget(self.btn_start)

        self.btn_swap = QPushButton("三手交换")
        self.btn_swap.setStyleSheet(btn_style)
        self.btn_swap.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_swap.clicked.connect(self._do_swap)
        self.btn_swap.hide()
        panel_layout.addWidget(self.btn_swap)

        self.btn_no_swap = QPushButton("不交换")
        self.btn_no_swap.setStyleSheet(btn_style)
        self.btn_no_swap.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_no_swap.clicked.connect(self._do_not_swap)
        self.btn_no_swap.hide()
        panel_layout.addWidget(self.btn_no_swap)

        self.btn_new = QPushButton("重置对局")
        self.btn_new.setStyleSheet(btn_style)
        self.btn_new.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_new.clicked.connect(self._reset_ui)
        panel_layout.addWidget(self.btn_new)

        layout.addWidget(panel)

    # ── 辅助方法 ──
    def _show_msg(self, msg):
        self.status_label.setText(msg)

    def _update_realtime_clock(self):
        if self.phase in (GamePhase.NOT_STARTED, GamePhase.GAME_OVER):
            return
        now = time.time()
        if self.game_start_time:
            total = now - self.game_start_time
            gm, gs = divmod(total, 60)
            self.game_time_label.setText(
                f"游戏进行时间: {int(gm):02d}:{int(gs):02d}"
            )
        if self.current_move_start_time:
            curr = now - self.current_move_start_time
            cm, cs = divmod(curr, 60)
            lm, ls = divmod(self.last_move_time, 60)
            self.move_time_label.setText(
                f"上一手花费: {int(lm)}分{ls:.1f}秒\n"
                f"当前手花费: {int(cm)}分{cs:.1f}秒"
            )

    def _mark_turn_start(self):
        if self.current_move_start_time is not None:
            self.last_move_time = time.time() - self.current_move_start_time
        self.current_move_start_time = time.time()

    def _record_time(self):
        if self.human_start_time:
            self.human_total_time += time.time() - self.human_start_time
            self.human_start_time = None

    def _update_time_label(self):
        total_moves = len(self.board.states) if self.board else 0
        self.turn_label.setText(f"第 {total_moves} 手")
        ai_m, ai_s = divmod(self.ai_total_time, 60)
        hu_m, hu_s = divmod(self.human_total_time, 60)
        ai_name = self.config.get('myName', '悟空')
        hu_name = self.config.get('oppoName', '人类')
        self.time_label.setText(
            f"{ai_name}耗时: {int(ai_m)}分{ai_s:.1f}秒\n"
            f"{hu_name}耗时: {int(hu_m)}分{hu_s:.1f}秒"
        )

    def _refresh_board(self):
        self.board_widget.board = self.board
        self.board_widget.five_candidates = self.five_candidate_moves
        self.board_widget.allow_hover = (
            not self.ai_thinking
            and self.phase not in (GamePhase.GAME_OVER, GamePhase.WAIT_CHOOSE_FIVE)
        )
        self.board_widget.update()

    # ── 重置 ──
    def _reset_ui(self):
        self.phase = GamePhase.NOT_STARTED
        self.ai_thinking = False
        self.five_candidate_moves = []
        self.game_start_time = None
        self.current_move_start_time = None
        self.last_move_time = 0.0
        self._timer.stop()

        # 关闭并行 AI 的 NN 服务
        if self.ai and hasattr(self.ai, 'shutdown'):
            self.ai.shutdown()
        self.ai = None

        self._load_config()
        self._show_msg("由配置决定先手方，点击开始游戏")
        self.btn_start.show()
        self.btn_swap.hide()
        self.btn_no_swap.hide()
        self.turn_label.setText("")
        self.game_time_label.setText("")
        self.move_time_label.setText("")
        self.time_label.setText("")
        self.board_widget.board = None
        self.board_widget.five_candidates = []
        self.board_widget.update()

    # ── 开始游戏 ──
    def _start_game(self, ai_first):
        self.btn_start.hide()

        self.board = game.Board(
            ExternalProgramManager=self.external_board,
            hash_table_manager=self.hash_table_manager,
            width=15, height=15, n_in_row=5
        )
        self.board.init_board(start_player=0)
        self.board.AI_turn = 0 if ai_first else 1

        model_path = "best_model/current_policy_step_best.model"
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "错误", f"找不到策略模型: {model_path}")
            self._reset_ui()
            return

        self._show_msg("加载 AI 模型中...")
        QApplication.processEvents()

        # 使用并行 AI
        num_workers = self.config.get('numWorkers', 4)
        max_batch = self.config.get('maxBatchSize', 8)
        self.ai = players.AIPlayerParallel(
            model_path=model_path,
            c_puct=2, n_playout=400,
            is_selfplay=False,
            num_workers=num_workers,
            max_batch_size=max_batch,
        )
        self.ai.mcts_player.search_time = self.config.get("searchTime", 0.1)

        self.ai_total_time = 0.0
        self.human_total_time = 0.0
        self.phase = GamePhase.OPENING_3
        self.five_candidate_moves = []

        self.game_start_time = time.time()
        self.current_move_start_time = time.time()
        self.last_move_time = 0.0
        self._timer.start()

        if ai_first:
            self._ai_fixed_opening()
        else:
            self._show_msg("人类先手 (黑方)\n请在天元(H8)落第1子")
            self.human_start_time = time.time()
            self._refresh_board()
            self._mark_turn_start()

    def _ai_fixed_opening(self):
        name = self.config.get('myName', '悟空')
        self._show_msg(f"{name}先手，默认选择疏星局开局...")
        QApplication.processEvents()

        for move in [112, 113, 9 * 15 + 9]:
            self.board.do_move(move)
            self._refresh_board()
            QApplication.processEvents()
            QThread.msleep(400)

        self.phase = GamePhase.WAIT_SWAP
        self._show_msg("疏星局开局完毕。\n请问是否三手交换？")
        self.btn_swap.show()
        self.btn_no_swap.show()
        self._mark_turn_start()

    # ── 三手交换 ──
    def _do_swap(self):
        self.btn_swap.hide()
        self.btn_no_swap.hide()
        self._show_msg("你选择了交换！你现在执黑。请为白方落第4子")
        self.board.AI_turn = 1 - self.board.AI_turn
        self.phase = GamePhase.NORMAL
        self.human_start_time = time.time()
        self._mark_turn_start()

    def _do_not_swap(self):
        self.btn_swap.hide()
        self.btn_no_swap.hide()
        self._show_msg("继续游戏（你不交换）。\n请落第4子")
        self.phase = GamePhase.NORMAL
        self.human_start_time = time.time()
        self._mark_turn_start()

    # ── 棋盘点击 ──
    @pyqtSlot(int, int)
    def _on_click(self, row, col):
        if self.phase in (GamePhase.NOT_STARTED, GamePhase.GAME_OVER):
            return
        if self.ai_thinking:
            return

        move = self.board.location_to_move((row, col))
        if move == -1:
            return

        if self.phase == GamePhase.OPENING_3:
            if move not in self.board.get_available(self.board.current_player):
                QMessageBox.warning(self, "非法", "此处无法落子或禁手")
                return
            if self.board.turn == 0 and move != 7 * 15 + 7:
                QMessageBox.warning(self, "规则", "第一手必须下在天元星(H8)")
                return
            self._human_do_move(move)
            if self.board.turn == 3:
                keys = list(self.board.states.keys())[:3]
                key_str = f"{keys[0]} {keys[1]} {keys[2]}"
                if key_str in openings_dict:
                    self.opening_info = openings_dict[key_str]
                    self.phase = GamePhase.WAIT_SWAP
                    self._show_msg(
                        f"局面: {self.opening_info[0]}\nAI 正在决定是否交换..."
                    )
                    self._refresh_board()
                    self._mark_turn_start()
                    QTimer.singleShot(1000, self._ai_decide_swap)
                else:
                    QMessageBox.critical(self, "错误", "非标准开局库开局，请重新开始")
                    self._reset_ui()

        elif self.phase == GamePhase.WAIT_FIVE_STRIKE_HUMAN:
            if move not in self.board.get_available(
                    self.board.current_player):
                return
            if move in self.five_candidate_moves:
                return
            self.five_candidate_moves.append(move)
            self._refresh_board()
            if len(self.five_candidate_moves) == self.five_nums:
                self.phase = GamePhase.WAIT_CHOOSE_FIVE
                self._show_msg(
                    f"你已给出 {self.five_nums} 个点。\nAI 正在选择最优打点..."
                )
                QApplication.processEvents()
                self._mark_turn_start()
                QTimer.singleShot(500, self._ai_choose_five_strike)
            else:
                remaining = self.five_nums - len(self.five_candidate_moves)
                self._show_msg(f"请再选择 {remaining} 个五手打点")

        elif self.phase == GamePhase.WAIT_CHOOSE_FIVE:
            if self.board.current_player != self.board.AI_turn:
                return
            if (move not in self.five_candidate_moves
                    and not any(move == int(m) for m in
                                self.five_candidate_moves)):
                QMessageBox.warning(self, "提示", "必须选择蓝色提示圈内的打点保留！")
                return
            self.board.do_move(move)
            self.five_candidate_moves = []
            self.phase = GamePhase.NORMAL
            self._show_msg("已保留打点。AI 思考中...")
            self._refresh_board()
            self._record_time()
            self._ai_launch_thread()
            self._mark_turn_start()

        elif self.phase == GamePhase.NORMAL:
            if move not in self.board.get_available(
                    self.board.current_player):
                QMessageBox.warning(self, "提示", "黑棋禁手或此处已有棋子")
                return
            self._human_do_move(move)

    def _ai_decide_swap(self):
        nums, ok = QInputDialog.getInt(
            self, "五手N打", "请输入五手打点数N (1-10)：",
            value=2, min=1, max=10
        )
        self.five_nums = nums if ok else 2
        info = self.opening_info
        name = self.config.get('myName', '悟空')
        if self.five_nums <= info[2]:
            self._show_msg(
                f"{info[0]}\n{name}选择【三手交换】！\n请你下第4子(白方)"
            )
            self.board.AI_turn = 1 - self.board.AI_turn
            self.phase = GamePhase.NORMAL
            self.human_start_time = time.time()
            self._mark_turn_start()
        else:
            self._show_msg(f"{info[0]}\n{name}【不交换】！\n{name}长考第4子...")
            self.phase = GamePhase.NORMAL
            QApplication.processEvents()
            self._ai_launch_thread()
            self._mark_turn_start()

    def _human_do_move(self, move):
        self._record_time()
        ok = self.board.do_move(move)
        if not ok:
            QMessageBox.critical(self, "犯规", "底层引擎报犯规！")
        self._refresh_board()
        self._update_time_label()
        if self._check_game_end():
            return

        if self.board.turn == 3:
            pass
        elif (self.board.turn == 4
              and self.board.current_player == self.board.AI_turn):
            self.phase = GamePhase.WAIT_FIVE_STRIKE_AI
            self._show_msg(f"AI 正在计算 {self.five_nums} 个五手打点...")
            self._ai_launch_thread(strike_5=True)
            self._mark_turn_start()
        elif (self.board.turn == 4
              and self.board.current_player != self.board.AI_turn):
            self.phase = GamePhase.WAIT_FIVE_STRIKE_HUMAN
            self._show_msg(f"需提供 {self.five_nums} 个打点，请在棋盘上连续点击")
            self.human_start_time = time.time()
            self._mark_turn_start()
        else:
            self.phase = GamePhase.NORMAL
            self._ai_launch_thread()
            self._mark_turn_start()

    # ── AI 五手选点 (QThread) ──
    def _ai_choose_five_strike(self):
        self.ai_thinking = True
        self._refresh_board()
        self.board_widget.repaint()
        self.status_label.repaint()
        self._ai_choose_worker = AIChooseWorkerThread(
            self.ai, self.board, self.five_candidate_moves
        )
        self._ai_choose_worker.finished.connect(self._on_ai_choose_done)
        QApplication.processEvents()
        self._ai_choose_worker.start()

    @pyqtSlot(int)
    def _on_ai_choose_done(self, best_move):
        self.ai_thinking = False
        self._show_msg("AI 已选择打点保留。AI 继续思考下第6手...")
        self.five_candidate_moves = []
        self.board.do_move(best_move)
        self.phase = GamePhase.NORMAL
        self._refresh_board()
        if self._check_game_end():
            return
        self._ai_launch_thread()
        self._mark_turn_start()

    # ── AI 主搜索 (QThread) ──
    def _ai_launch_thread(self, strike_5=False):
        self.ai_thinking = True
        self._refresh_board()
        self.board_widget.repaint()  # 强制同步重绘落子
        
        if not strike_5:
            name = self.config.get('myName', '悟空')
            self._show_msg(f"{name}长考中... (并行 MCTS)")
        
        self.btn_swap.hide()
        self.btn_no_swap.hide()
        self.status_label.repaint()  # 强制同步刷新状态文本

        self._ai_start_time = time.time()
        self._ai_worker = AIWorkerThread(
            self.ai, self.board, strike_5=strike_5
        )
        if strike_5:
            self._ai_worker._five_nums = self.five_nums
        self._ai_worker.finished.connect(self._on_ai_done)
        QApplication.processEvents()
        self._ai_worker.start()

    @pyqtSlot(str, object)
    def _on_ai_done(self, status, data):
        cost = time.time() - self._ai_start_time
        self.ai_total_time += cost
        self.ai_thinking = False

        if status == 'error':
            self._show_msg(f"AI 故障: {data}")
            return

        if status == 'strike_5':
            self.five_candidate_moves = data
            self.phase = GamePhase.WAIT_CHOOSE_FIVE
            self._show_msg(
                "AI 提出了五手打点候选。\n"
                "请点击蓝色提示圈，选择你要留给 AI 的落子。"
            )
            self._refresh_board()
            self.human_start_time = time.time()
            self._mark_turn_start()
        else:
            move = data
            self.board.do_move(move)
            self._refresh_board()
            self._update_time_label()
            if self._check_game_end():
                return

            if (self.board.turn == 4
                    and self.board.current_player != self.board.AI_turn):
                self.phase = GamePhase.WAIT_FIVE_STRIKE_HUMAN
                self._show_msg(
                    f"请你连续点击棋盘，提供 {self.five_nums} 个五手打点"
                )
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
        self._timer.stop()
        self._refresh_board()

        if winner == -1:
            msg = "平局！棋盘已满。"
        elif self.board.have_non_compliance == 1:
            loser = "黑棋" if self.board.history_player == 0 else "白棋"
            msg = f"{loser} 犯规（长连/四四/三三禁手）判负！\n对方获胜。"
        else:
            ai_name = self.config.get('myName', '悟空')
            hu_name = self.config.get('oppoName', '你(人类)')
            who = (f"{ai_name}(AI)"
                   if winner == self.board.AI_turn
                   else f"{hu_name}")
            msg = f"游戏结束！{who} 连成五子获胜！"

        self._show_msg(msg)
        # 避免通过 lambda 捕获 msg 造成变量丢失或 GC 问题
        QTimer.singleShot(300, lambda: self._show_game_over_dialog(msg))
        return True

    def _show_game_over_dialog(self, msg):
        QApplication.processEvents()
        QMessageBox.information(self, "对局结束", msg)

    # ── 关闭事件 ──
    def closeEvent(self, event):
        if self.ai and hasattr(self.ai, 'shutdown'):
            self.ai.shutdown()
        event.accept()


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print(" 悟空五子棋竞技版 (PyQt5 并发): MCTS + TSS + NNBatch")
    print("=" * 60)

    if not os.path.exists("merged_hash_new2.pkl"):
        print("[提示] 找不到 merged_hash_new2.pkl，将被自动创建空哈希表。")

    externalProgramManager = gomoku_engine.Board()
    hash_table_manager = game.HashTableManager("merged_hash_new2.pkl")

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))

    window = GomokuMainWindow(externalProgramManager, hash_table_manager)
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
