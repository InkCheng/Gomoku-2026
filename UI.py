from datetime import datetime
import gomoku_engine
import pygame as pg
import os
import sys
import time
import src.game as game
import src.players as players
import heapq

WIDTH, HEIGHT = 800, 615

# 统一的关闭函数
def quit_game():
    pg.quit()
    sys.exit()
RED = (255, 255, 255)
BLUE = (255, 255, 255)
GREEN = (255, 255, 255)


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

def get_wisemove(board):
    res=[]
    for pos in board.states.keys():
        for i in range(-3, 4):
            for j in range(-3, 4):
                    new_pos = pos + i * board.width + j
                    if 0 <= new_pos < board.width * board.height and new_pos not in res and new_pos not in board.states.keys() and new_pos in board.availables:
                        res.append(new_pos)
    return res

def build_max_heap(arr):
    # 将所有元素取反以构建最大堆
    max_heap = [-elem for elem in arr]
    heapq.heapify(max_heap)
    # 取反回来恢复原始值
    max_heap = [-elem for elem in max_heap]
    return max_heap


class GameObject:
    # 具有棋子的图像、类别和坐标三个属性
    def __init__(self, image, color, pos):
        self.image = image
        self.color = color
        self.pos = image.get_rect(center=pos)


# 按钮类，生成了悔棋按钮和恢复按钮
class Button(object):
    # 具有图像surface，宽高和坐标属性
    def __init__(self, text, color, x=None, y=None):
        self.surface = font_big.render(text, True, color)
        self.WIDTH = self.surface.get_width()
        self.HEIGHT = self.surface.get_height()
        self.x = x
        self.y = y

    # 这个方法用于确定鼠标是否点击了对应的按钮
    def check_click(self, position):
        x_match = self.x < position[0] < self.x + self.WIDTH
        y_match = self.y < position[1] < self.y + self.HEIGHT
        if x_match and y_match:
            return True
        else:
            return False


# 若落子位置已经有棋子，则进行提示

def main(board_inner,AI_player):
    AI_total_time = 0
    human_total_time = 0
    symmetric_flags = None
    begin_time = time.time()

    pg.init()
    # 一系列数据初始化
    clock = pg.time.Clock()  # pygame时钟

    objects = []  # 下棋记录列表
    recover_objects = []  # 恢复棋子时用到的列表，即悔棋记录列表
    ob_list = [objects, recover_objects]  # 将以上两个列表放到一个列表中，主要是增强抽象度，简少了代码行数

    screen = pg.display.set_mode((WIDTH, HEIGHT))  # 游戏窗口

    black = pg.image.load("data/chess_black.png").convert_alpha()  # 黑棋棋子图像
    white = pg.image.load("data/chess_white.png").convert_alpha()  # 白棋棋子图像
    black_temp=pg.image.load("data/chess_black_temp.png").convert_alpha()  # 黑棋棋子图像
    red_dot = pg.image.load("data/red_dot.png").convert_alpha()  # 黑棋棋子图像

    background = pg.image.load("data/bg_old.png").convert_alpha()  # 棋盘背景图像
    regret_button = Button('悔棋', RED, 665, 200)  # 创建悔棋按钮
    recover_button = Button('恢复', BLUE, 665, 300)  # 创建恢复按钮
    change_button = Button('三手交换', GREEN, 625, 400)  # 交换按钮
    continue_button = Button('继续游戏', GREEN, 625, 500)  # 交换按钮
    AI_first_button = Button('悟空先手', GREEN, 625, 100)  # AI先手按钮
    human_first_button = Button('人类先手', GREEN, 625, 50)  # 人类先手按钮

    screen.blit(regret_button.surface, (regret_button.x, regret_button.y))  # 把悔棋按钮打印游戏窗口
    screen.blit(recover_button.surface, (recover_button.x, recover_button.y))  # 把恢复按钮打印游戏窗口
    screen.blit(change_button.surface, (change_button.x, change_button.y))  # 把交换按钮打印游戏窗口
    screen.blit(AI_first_button.surface, (AI_first_button.x, AI_first_button.y))  # 把AI先手按钮打印游戏窗口
    screen.blit(human_first_button.surface, (human_first_button.x, human_first_button.y))  # 把人类先手按钮打印游戏窗口
    screen.blit(continue_button.surface, (continue_button.x, continue_button.y))  # 把继续按钮打印游戏窗口


    pg.display.set_caption("悟空五子棋")  # 窗体的标题



    going = True  # 主循环变量，用于控制主循环继续或者结束
    chess_list = [ black,white,black_temp,red_dot]  # 棋子图像列表，主要是增强抽象度，简少了代码行数
    letter_list = ['X','O' ]  # 棋子类型列表，主要是增强抽象度，简少了代码行数
    word_list = [ '黑棋','白棋']  # 棋子文字名称列表，主要是增强抽象度，简少了代码行数
    word_list_temp = ["对方","悟空"]

    word_color = [ (0, 0, 0),(255, 255, 255)]  # 棋子文字颜色列表，主要是增强抽象度，简少了代码行数
    #需要指定局是谁，0是AI，1是人

    first_player_chosen = False
    start_player = 0  # 先手玩家，0表示AI先手，1表示人类先手
    five_nums=2  #五打n手
    swap_flag=False  #是否可以交换
    min_heap = []  # 选择最差点的堆


    while not first_player_chosen:
        screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
        # 在选择先手之前，显示提示信息
        hint_text = font.render("请选择先手", True, RED)
        hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=100)
        screen.blit(hint_text, hint_text_pos)
        pg.display.update()

        for event in pg.event.get():
            if event.type == pg.QUIT:
                quit_game()
            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                quit_game()
            elif event.type == pg.MOUSEBUTTONDOWN:
                pos = pg.mouse.get_pos()
                if AI_first_button.check_click(pos):
                    start_player = 0
                    board_inner.init_board(start_player=start_player)
                    first_player_chosen = True
                elif human_first_button.check_click(pos):
                    start_player = 1
                    board_inner.init_board(start_player=start_player)
                    first_player_chosen = True
                else:
                    hint_text = font.render("请选择先手", True, RED)
                    hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=100)
                    screen.blit(hint_text, hint_text_pos)
                    pg.display.update()

    if start_player == 0:
        hint_text = font.render("悟空先手", True, word_color[0])  # 提示文案
        hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=200)  # 提示文案位置
        screen.blit(hint_text, hint_text_pos)
        pg.display.update()
        pg.time.delay(1000)  # 暂停1秒，保证文案能够清晰展示

        board_list = [112, 113, 9 * 15 + 9]
        screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
        for i in range(3):
            x, y = board_inner.move_to_location(board_list[i])
            board_inner.do_move(board_list[i])
            objects.append(GameObject(chess_list[board_inner.current_player == board_inner.start_player],
                                      letter_list[board_inner.current_player == board_inner.start_player],
                                      (27 + x * 40, 27 + (14 - y) * 40))
                           )

        # 将下棋记录的棋子打印到游戏窗口
        for o in objects:
            screen.blit(o.image, o.pos)
        print(board_inner.states)
        hint_text = font.render("悟空选择疏星局开局,选择五手二打", True, word_color[0])  # 提示文案
        hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=100)  # 提示文案位置
        print("悟空选择疏星局开局,选择五手二打")
        screen.blit(hint_text, hint_text_pos)



        five_nums=2
        pg.display.update()
        pg.time.delay(2000)  # 暂停1秒，保证文案能够清晰展示
        flag = False
        while not swap_flag:
            screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
            # 在选择先手之前，显示提示信息
            hint_text = font.render("请选择是否三手交换", True, RED)
            hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=100)
            screen.blit(hint_text, hint_text_pos)
            for o in objects:
                screen.blit(o.image, o.pos)
            # print(board_inner.states)
            pg.display.update()

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    quit_game()
                elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    quit_game()
                elif event.type == pg.MOUSEBUTTONDOWN:
                    pos = pg.mouse.get_pos()
                    if change_button.check_click(pos):
                        swap_flag=True
                        board_inner.AI_turn=1-board_inner.AI_turn
                        flag=True
                    elif continue_button.check_click(pos):
                        swap_flag=True

                    else:
                        hint_text = font.render("请选择是否三手交换，或者继续游戏", True, RED)
                        hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=100)
                        screen.blit(hint_text, hint_text_pos)
                        pg.display.update()

        if flag:
            x, y = board_inner.move_to_location(8*15+8)
            board_inner.do_move(8*15+8)
            objects.append(GameObject(chess_list[board_inner.current_player == board_inner.start_player],
                                      letter_list[board_inner.current_player == board_inner.start_player],
                                      (27 + x * 40, 27 + (14 - y) * 40))
                           )
            for o in objects:
                screen.blit(o.image, o.pos)
            # print(board_inner.states)
            pg.display.update()

            hint_text = font.render("请你给出五手二打的落子位置", True, word_color[0])  # 提示文案
            hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=200)  # 提示文案位置
            screen.blit(hint_text, hint_text_pos)
            pg.display.update()

            cnt = 0
            while cnt < five_nums:
                screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
                text = font.render("第{}手".format(cnt + 1),
                                   True, word_color[
                                       board_inner.current_player == board_inner.start_player])  # 创建一个文本对象，显示当前是哪方的回合
                text_pos = text.get_rect(centerx=background.get_width() / 2, y=2)  # 确定文本对象的显示位置
                screen.blit(text, text_pos)  # 将文本对象打印到游戏窗口
                for o in objects:
                    screen.blit(o.image, o.pos)
                # print(board_inner.states)
                pg.display.update()
                for event in pg.event.get():
                    # 如果关闭窗口，主循环结束
                    if event.type == pg.QUIT:
                        quit_game()
                    # 如果点击键盘ESC键，主循环结束
                    elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                        quit_game()
                    # 如果玩家进行了鼠标点击操作
                    elif event.type == pg.MOUSEBUTTONDOWN:
                        pos = pg.mouse.get_pos()  # 获取鼠标点击坐标

                        # 若用户点击的不是悔棋、恢复按钮，则进行落子操作
                        a, b = round((pos[0] - 27) / 40), round((pos[1] - 27) / 40)  # 将用户鼠标点击位置的坐标，换算为board坐标
                        # 若坐标非法（即点击到了黑色区域），则不做处理
                        if a >= 15 or b >= 15:
                            continue
                        else:
                            x, y = max(0, a) if a < 0 else min(a, 14), max(0, b) if b < 0 else min(b, 14)  # 将a、b进行处理得到x和y
                            print('*' * 50)
                            print("棋盘1", x, 14 - y)
                            print("棋盘2:", x, y)
                            print("棋盘3：",chr(ord('A')+x),15-y)
                            print('*' * 50)
                            print()
                            # 判断落子位置是否合法
                            # 若落子操作合法，则进行落子
                            move=board_inner.location_to_move((x, 14 - y))
                            if move in board_inner.availables:
                                board_inner.do_move(move)
                                objects.append(
                                    GameObject(chess_list[2],
                                               letter_list[0],
                                               (27 + x * 40, 27 + y * 40)))
                                # 一旦成功落子，则将悔棋记录列表清空；不这么做，一旦在悔棋和恢复中间掺杂落子操作，就会有问题
                                pro, value = AI_player.evaluate(board_inner)
                                board_inner.undo_move(x, 14 - y)
                                heapq.heappush(min_heap, (-value,move))
                                cnt+=1

                            else:
                                hint_text = font.render("该位置已有棋子", True, word_color[0])  # 提示文案
                                hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=200)  # 提示文案位置
                                # 将下棋记录的棋子打印到游戏窗口
                            for o in objects:
                                screen.blit(o.image, o.pos)
                            screen.blit(hint_text, hint_text_pos)  # 把提示文案打印到游戏窗口
                            print(board_inner.states)
                            pg.display.update()  # 对游戏窗口进行刷新
                            pg.time.delay(300)  # 暂停0.3秒，保证文案能够清晰展示\


            for o in objects:
                screen.blit(o.image, o.pos)
            pg.display.update()
            print(board_inner.states)

            for i in range(five_nums):
                objects.pop()

            # 选择最差点的落子
            if min_heap:
                value,move= heapq.heappop(min_heap)
                value = -value
                x, y = board_inner.move_to_location(move)
                board_inner.do_move(move)
                # print(board_inner.move_to_location(move),"        ",value)
                objects.append(GameObject(chess_list[board_inner.current_player == board_inner.start_player], letter_list[board_inner.current_player == board_inner.start_player], (27 + x * 40, 27 + (14 - y) * 40)))
                screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
                for o in objects:
                    screen.blit(o.image, o.pos)
                pg.display.update()

            print(board_inner.states)

        else:
            hint_text = font.render("继续游戏，悟空保持先手，请你落子", True, word_color[0])  # 提示文案
            screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
            hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=100)  # 提示文案位置
            screen.blit(hint_text, hint_text_pos)
            for o in objects:
                screen.blit(o.image, o.pos)

            pg.display.update()

            pg.time.delay(1000)  # 暂停1秒，保证文案能够清晰展示
            flag=0
            temp_F=0
            while flag==0:
                for o in objects:
                    screen.blit(o.image, o.pos)

                pg.display.update()
                for event in pg.event.get():
                    # 如果关闭窗口，主循环结束
                    if event.type == pg.QUIT:
                        quit_game()
                    # 如果点击键盘ESC键，主循环结束
                    elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                        quit_game()
                    # 如果玩家进行了鼠标点击操作
                    elif event.type == pg.MOUSEBUTTONDOWN:
                        pos = pg.mouse.get_pos()  # 获取鼠标点击坐标

                        # 若用户点击的不是悔棋、恢复按钮，则进行落子操作
                        a, b = round((pos[0] - 27) / 40), round((pos[1] - 27) / 40)  # 将用户鼠标点击位置的坐标，换算为board坐标
                        # 若坐标非法（即点击到了黑色区域），则不做处理
                        if a >= 15 or b >= 15:
                            continue
                        else:
                            x, y = max(0, a) if a < 0 else min(a, 14), max(0, b) if b < 0 else min(b, 14)  # 将a、b进行处理得到x和y
                            print('*' * 50)
                            print("棋盘1", x, 14 - y)
                            print("棋盘2:", x, y)
                            print("棋盘3：", chr(ord('A') + x), 15 - y)
                            print('*' * 50)
                            print()
                            # 判断落子位置是否合法
                            # 若落子操作合法，则进行落子

                            if board_inner.location_to_move((x, 14 - y)) in board_inner.availables:
                                flag=1
                                move=board_inner.location_to_move((x, 14 - y))
                                if move==128:
                                    temp_F=1
                                board_inner.do_move(move)
                                objects.append(
                                    GameObject(chess_list[board_inner.current_player == board_inner.start_player],
                                               letter_list[board_inner.current_player == board_inner.start_player],
                                               (27 + x * 40, 27 + y * 40)))
                                # 一旦成功落子，则将悔棋记录列表清空；不这么做，一旦在悔棋和恢复中间掺杂落子操作，就会有问题
                                recover_objects.clear()

                            else:
                                board_inner.get_available()
                                hint_text = font.render("该位置落子违法", True, word_color[0])  # 提示文案
                                hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=200)  # 提示文案位置
                                # 将下棋记录的棋子打印到游戏窗口
                                for o in objects:
                                    screen.blit(o.image, o.pos)

                                screen.blit(hint_text, hint_text_pos)  # 把提示文案打印到游戏窗口
                                pg.display.update()  # 对游戏窗口进行刷新
                                pg.time.delay(300)  # 暂停0.3秒，保证文案能够清晰展示


            screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
            for o in objects:
                screen.blit(o.image, o.pos)

            pg.display.update()

            if temp_F:
                moves = [6*15+8,9*15+8]
                objects.append(
                    GameObject(chess_list[2],
                               letter_list[0],
                               (27 + 6* 40, 27 + (14 - 8) * 40)))
                objects.append(
                    GameObject(chess_list[2],
                               letter_list[0],
                               (27 + 9 * 40, 27 + (14 - 8) * 40)))
            else:
                wise_move = get_wisemove(board_inner)
                min_leaf_heap = []  # 选择最差点的堆
                for move in wise_move:
                    location = board.move_to_location(move)
                    board.do_move(move)
                    action_probs, leaf_value = AI_player.evaluate(board)
                    x, y = board.move_to_location(move)
                    board.undo_move(x, y)
                    heapq.heappush(min_leaf_heap, (leaf_value,move ))
                moves= []
                for i in range(five_nums):
                    if min_leaf_heap:
                        value,move = heapq.heappop(min_leaf_heap)
                        moves.append(move)
                        x, y = board_inner.move_to_location(move)
                        # print(board_inner.move_to_location(move), "        ", value)
                        objects.append(
                            GameObject(chess_list[2],
                                       letter_list[0],
                                       (27 + x * 40, 27 + (14-y) * 40)))


            hint_text = font.render("悟空已经完成打点，请你选择", True, word_color[0])  # 提示文案
            screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
            hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=200)  # 提示文案位置
            screen.blit(hint_text, hint_text_pos)

            for o in objects:
                screen.blit(o.image, o.pos)

            pg.display.update()
            pg.time.delay(1000)  # 暂停1秒，保证文案能够清晰展示

            flag = 0

            while flag == 0:

                screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
                for o in objects:
                    screen.blit(o.image, o.pos)

                pg.display.update()
                for event in pg.event.get():
                    # 如果关闭窗口，主循环结束
                    if event.type == pg.QUIT:
                        quit_game()
                    # 如果点击键盘ESC键，主循环结束
                    elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                        quit_game()
                    # 如果玩家进行了鼠标点击操作
                    elif event.type == pg.MOUSEBUTTONDOWN:
                        pos = pg.mouse.get_pos()  # 获取鼠标点击坐标

                        # 若用户点击的不是悔棋、恢复按钮，则进行落子操作
                        a, b = round((pos[0] - 27) / 40), round((pos[1] - 27) / 40)  # 将用户鼠标点击位置的坐标，换算为board坐标
                        # 若坐标非法（即点击到了黑色区域），则不做处理
                        if a >= 15 or b >= 15:
                            continue
                        else:
                            x, y = max(0, a) if a < 0 else min(a, 14), max(0, b) if b < 0 else min(b,14)  # 将a、b进行处理得到x和y
                            print('*' * 50)
                            print("棋盘1", x, 14 - y)
                            print("棋盘2:", x, y)
                            print("棋盘3：", chr(ord('A') + x), 15 - y)
                            print('*' * 50)
                            print()
                            # 判断落子位置是否合法
                            # 若落子操作合法，则进行落子

                            if board_inner.location_to_move((x, 14 - y)) in moves:
                                flag = 1
                                for i in range(five_nums):
                                    objects.pop()
                                board_inner.do_move(board_inner.location_to_move((x, 14 - y)))
                                objects.append(
                                    GameObject(chess_list[board_inner.current_player == board_inner.start_player],
                                               letter_list[board_inner.current_player == board_inner.start_player],
                                               (27 + x * 40, 27 + y * 40)))
                                # 一旦成功落子，则将悔棋记录列表清空；不这么做，一旦在悔棋和恢复中间掺杂落子操作，就会有问题
                                recover_objects.clear()

                            else:
                                hint_text = font.render("该位置落子不属于打点", True, word_color[0])  # 提示文案
                                hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=200)  # 提示文案位置
                                # 将下棋记录的棋子打印到游戏窗口
                                for o in objects:
                                    screen.blit(o.image, o.pos)

                                screen.blit(hint_text, hint_text_pos)  # 把提示文案打印到游戏窗口
                                pg.display.update()  # 对游戏窗口进行刷新
                                pg.time.delay(300)  # 暂停0.3秒，保证文案能够清晰展示



# TODO：实现人类先手功能

    else:
        flag = 0
        again_flag = 0
        while flag == 0:
                if again_flag:
                    board_inner.init_board(start_player=start_player)
                    objects.clear()
                    ob_list.clear()

                hint_text = font.render("人类先手，请你落子开局", True, word_color[1])  # 提示文案
                hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=200)  # 提示文案位置
                screen.blit(hint_text, hint_text_pos)
                pg.display.update()

                pg.time.delay(1000)  # 暂停1秒，保证文案能够清晰展示

                cnt = 0
                while cnt<3:

                    screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
                    text = font.render("第{}落子".format(cnt + 1),
                                       True, word_color[
                                           board_inner.current_player != board_inner.start_player])  # 创建一个文本对象，显示当前是哪方的回合
                    text_pos = text.get_rect(centerx=background.get_width() / 2, y=2)  # 确定文本对象的显示位置
                    screen.blit(text, text_pos)  # 将文本对象打印到游戏窗口
                    for o in objects:
                        screen.blit(o.image, o.pos)

                    pg.display.update()

                    for event in pg.event.get():
                        # 如果关闭窗口，主循环结束
                        if event.type == pg.QUIT:
                            quit_game()
                        # 如果点击键盘ESC键，主循环结束
                        elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                            quit_game()

                        # 如果玩家进行了鼠标点击操作
                        elif event.type == pg.MOUSEBUTTONDOWN:
                            pos = pg.mouse.get_pos()  # 获取鼠标点击坐标
                            if regret_button.check_click(pos):
                                if cnt > 0:
                                    cnt -= 1
                                    x, y = [round((p + 18 - 27) / 40) for p in ob_list[0][-1].pos[:2]]

                                    print('*' * 50)
                                    print("撤回棋子：")
                                    print("棋盘1：", x, 14 - y)
                                    print("棋盘2：", x, y)
                                    print("棋盘3：", chr(ord('A') + x), 15 - y)
                                    print('*' * 50)
                                    print()
                                    # 如果是悔棋操作，则board指定元素值恢复为' '；如果是恢复操作，则指定坐标board指定元素重新赋值
                                    board_inner.undo_move(x, 14 - y)
                                    ob_list[0 - 1].append(ob_list[0][-1])  # 将游戏/悔棋记录列表的最后一个值添加到悔棋/下棋记录列表
                                    ob_list[0].pop()  # 将游戏/悔棋记录列表的最后一个值删除
                                    print(board_inner.states)
                            else:
                                # 若用户点击的不是悔棋、恢复按钮，则进行落子操作
                                a, b = round((pos[0] - 27) / 40), round((pos[1] - 27) / 40)  # 将用户鼠标点击位置的坐标，换算为board坐标
                                # 若坐标非法（即点击到了黑色区域），则不做处理
                                if a >= 15 or b >= 15:
                                    continue
                                else:
                                    x, y = max(0, a) if a < 0 else min(a, 14), max(0, b) if b < 0 else min(b, 14)  # 将a、b进行处理得到x和y
                                    print('*' * 50)
                                    print("棋盘1", x, 14 - y)
                                    print("棋盘2:", x, y)
                                    print("棋盘3：", chr(ord('A') + x), 15 - y)
                                    print('*' * 50)
                                    print()
                                    # 判断落子位置是否合法
                                    # 若落子操作合法，则进行落子
                                    if board_inner.location_to_move((x, 14 - y)) in board_inner.availables:
                                        cnt+=1
                                        board_inner.do_move(board_inner.location_to_move((x, 14 - y)))
                                        objects.append(
                                            GameObject(chess_list[board_inner.current_player == board_inner.start_player],
                                                       letter_list[board_inner.current_player == board_inner.start_player],
                                                       (27 + x * 40, 27 + y * 40)))
                                        # 一旦成功落子，则将悔棋记录列表清空；不这么做，一旦在悔棋和恢复中间掺杂落子操作，就会有问题
                                        recover_objects.clear()

                                    else:
                                        hint_text = font.render("该位置落子违法", True, word_color[0])  # 提示文案
                                        hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=200)  # 提示文案位置
                                        # 将下棋记录的棋子打印到游戏窗口
                                        for o in objects:
                                            screen.blit(o.image, o.pos)

                                        screen.blit(hint_text, hint_text_pos)  # 把提示文案打印到游戏窗口
                                        pg.display.update()  # 对游戏窗口进行刷新
                                        pg.time.delay(300)  # 暂停0.3秒，保证文案能够清晰展示

                screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
                for o in objects:
                    screen.blit(o.image, o.pos)
                pg.display.update()

                first_three_moves = list(board_inner.states.keys())[:3]
                # 将前三个move用空格连接成字符串
                result = ' '.join(map(str, first_three_moves))
                # 判断是否在开局库中
                if result in openings_dict:
                    hint_text = font.render("请你输入五打n手数量", True, word_color[1])  # 提示文案
                    hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=200)  # 提示文案位置
                    screen.blit(hint_text, hint_text_pos)
                    pg.display.update()
                    for o in objects:
                        screen.blit(o.image, o.pos)

                    pg.display.update()

                    # 等待一秒，让用户先看到棋盘状态
                    pg.time.delay(1000)

                    # 使用GUI方式获取用户输入，避免阻塞
                    five_nums = 0
                    input_text = ""
                    typing_mode = True

                    while typing_mode:
                        screen.blit(background, (0, 0))

                        # 先绘制棋子
                        for o in objects:
                            screen.blit(o.image, o.pos)

                        # 绘制白色弹窗背景
                        dialog_width = 400
                        dialog_height = 300
                        dialog_x = (WIDTH - dialog_width) // 2
                        dialog_y = (HEIGHT - dialog_height) // 2

                        # 绘制弹窗阴影
                        pg.draw.rect(screen, (100, 100, 100),
                                   (dialog_x + 5, dialog_y + 5, dialog_width, dialog_height), border_radius=10)
                        # 绘制白色弹窗
                        pg.draw.rect(screen, (255, 255, 255),
                                   (dialog_x, dialog_y, dialog_width, dialog_height), border_radius=10)
                        # 绘制弹窗边框
                        pg.draw.rect(screen, (50, 50, 50),
                                   (dialog_x, dialog_y, dialog_width, dialog_height), 2, border_radius=10)

                        # 绘制弹窗标题
                        title_text = font.render("请输入五打n手数量", True, (0, 0, 0))
                        title_pos = title_text.get_rect(centerx=WIDTH // 2, y=dialog_y + 30)
                        screen.blit(title_text, title_pos)

                        # 绘制输入框
                        input_box_width = 200
                        input_box_height = 50
                        input_box_x = (WIDTH - input_box_width) // 2
                        input_box_y = dialog_y + 150

                        # 输入框背景
                        pg.draw.rect(screen, (240, 240, 240),
                                   (input_box_x, input_box_y, input_box_width, input_box_height), border_radius=5)
                        # 输入框边框
                        pg.draw.rect(screen, (0, 100, 200),
                                   (input_box_x, input_box_y, input_box_width, input_box_height), 2, border_radius=5)

                        # 显示输入的数字
                        input_display = font_big.render(input_text, True, (0, 0, 0))
                        input_pos = input_display.get_rect(center=(WIDTH // 2, input_box_y + input_box_height // 2))
                        screen.blit(input_display, input_pos)

                        # 显示操作提示
                        hint_text = font.render("按回车确认，按ESC取消", True, (150, 150, 150))
                        hint_pos = hint_text.get_rect(centerx=WIDTH // 2, y=dialog_y + 240)
                        screen.blit(hint_text, hint_pos)

                        pg.display.update()

                        for event in pg.event.get():
                            if event.type == pg.QUIT:
                                quit_game()
                            elif event.type == pg.KEYDOWN:
                                if event.key == pg.K_ESCAPE:
                                    quit_game()
                                elif event.key == pg.K_RETURN:
                                    if input_text:
                                        five_nums = int(input_text)
                                        if 2 <= five_nums <= 5:
                                            typing_mode = False
                                        else:
                                            input_text = ""  # 清空无效输入
                                elif event.key == pg.K_BACKSPACE:
                                    input_text = input_text[:-1]
                                elif event.unicode.isdigit():
                                    input_text += event.unicode

                    flag = 1

                    print(openings_dict[result][0])
                    print(openings_dict[result][2])

                    if five_nums<=int(openings_dict[result][2]):

                        screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
                        hint_text = font.render(f"该局面为{openings_dict[result][0]}，悟空选择三手交换", True, word_color[1])  # 提示文案
                        hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=100)  # 提示文案位置
                        screen.blit(hint_text, hint_text_pos)
                        board_inner.AI_turn=1-board_inner.AI_turn
                        hint_text_temp = font.render("请你进行落子", True, word_color[1])  # 提示文案
                        hint_text_pos_temp = hint_text.get_rect(centerx=background.get_width() / 2, y=200)  # 提示文案位置
                        screen.blit(hint_text_temp, hint_text_pos_temp)
                        print("悟空选择三手交换,请你落子")
                        #停留1秒
                        for o in objects:
                            screen.blit(o.image, o.pos)

                        pg.display.update()
                        pg.time.delay(1000)

                        temp_flag=0
                        while temp_flag==0:
                            for event in pg.event.get():
                                # 如果关闭窗口，主循环结束
                                if event.type == pg.QUIT:
                                    quit_game()
                                # 如果点击键盘ESC键，主循环结束
                                elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                                    quit_game()

                                # 如果玩家进行了鼠标点击操作
                                elif event.type == pg.MOUSEBUTTONDOWN:
                                    pos = pg.mouse.get_pos()  # 获取鼠标点击坐标
                                    # 若用户点击的不是悔棋、恢复按钮，则进行落子操作
                                    a, b = round((pos[0] - 27) / 40), round((pos[1] - 27) / 40)  # 将用户鼠标点击位置的坐标，换算为board坐标
                                    # 若坐标非法（即点击到了黑色区域），则不做处理
                                    if a >= 15 or b >= 15:
                                        continue
                                    else:
                                        x, y = max(0, a) if a < 0 else min(a, 14), max(0, b) if b < 0 else min(b,
                                                                                                               14)  # 将a、b进行处理得到x和y
                                        print('*' * 50)
                                        print("棋盘1", x, 14 - y)
                                        print("棋盘2:", x, y)
                                        print("棋盘3：", chr(ord('A') + x), 15 - y)
                                        print('*' * 50)
                                        print()
                                        # 判断落子位置是否合法
                                        # 若落子操作合法，则进行落子

                                        if board_inner.location_to_move((x, 14 - y)) in board_inner.availables:
                                            temp_flag=1
                                            board_inner.do_move(board_inner.location_to_move((x, 14 - y)))
                                            objects.append(
                                                GameObject(
                                                    chess_list[board_inner.current_player == board_inner.start_player],
                                                    letter_list[board_inner.current_player == board_inner.start_player],
                                                    (27 + x * 40, 27 + y * 40)))
                                            # 一旦成功落子，则将悔棋记录列表清空；不这么做，一旦在悔棋和恢复中间掺杂落子操作，就会有问题
                                            recover_objects.clear()

                                        else:
                                            board_inner.get_available()
                                            hint_text = font.render("该位置落子违法", True, word_color[0])  # 提示文案
                                            hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2,
                                                                               y=200)  # 提示文案位置
                                            # 将下棋记录的棋子打印到游戏窗口
                                            for o in objects:
                                                screen.blit(o.image, o.pos)

                                            screen.blit(hint_text, hint_text_pos)  # 把提示文案打印到游戏窗口
                                            pg.display.update()  # 对游戏窗口进行刷新
                                            pg.time.delay(300)  # 暂停0.3秒，保证文案能够清晰展示

                        screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
                        for o in objects:
                            screen.blit(o.image, o.pos)

                        pg.display.update()
                        first_three_moves = list(board_inner.states.keys())[:4]
                        # 将前三个move用空格连接成字符串
                        result = ' '.join(map(str, first_three_moves))
                        if result=="112 113 144 128":
                            moves = [6 * 15 + 8, 9 * 15 + 8]
                            objects.append(
                                GameObject(chess_list[2],
                                           letter_list[0],
                                           (27 + 6 * 40, 27 + (14 - 8) * 40)))
                            objects.append(
                                GameObject(chess_list[2],
                                           letter_list[0],
                                           (27 + 9 * 40, 27 + (14 - 8) * 40)))
                        elif result=="112 113 84 98":
                            moves = [8 * 15 + 8, 5 * 15 + 8]
                            objects.append(
                                GameObject(chess_list[2],
                                           letter_list[0],
                                           (27 + 8 * 40, 27 + (14 - 8) * 40)))
                            objects.append(
                                GameObject(chess_list[2],
                                           letter_list[0],
                                           (27 + 5 * 40, 27 + (14 - 8) * 40)))
                        else:
                            #AI开始打点
                            min_leaf_heap = []  # 选择最差点的堆
                            wise_moves=get_wisemove(board_inner)
                            for move in wise_moves:
                                board.do_move(move)
                                action_probs, leaf_value = AI_player.evaluate(board)
                                x, y = board.move_to_location(move)
                                board.undo_move(x, y)
                                heapq.heappush(min_leaf_heap, (leaf_value, move))
                            moves = []
                            temp_five_nums=five_nums
                            symmetric_flags = board_inner.is_symmetric()
                            while temp_five_nums > 0:
                                if min_leaf_heap:
                                    value, move = heapq.heappop(min_leaf_heap)
                                    row, col = board_inner.move_to_location(move)
                                    if symmetric_flags[0]:
                                        mirror_move = board_inner.location_to_move((board_inner.height - 1 - row, col))
                                        if moves.count(mirror_move) == 1:
                                            print("水平轴上出现对称点，跳过")
                                            continue
                                    if symmetric_flags[1]:
                                        mirror_move = board_inner.location_to_move((row, board_inner.width - 1 - col))
                                        if moves.count(mirror_move) == 1:
                                            print("垂直轴上出现对称点，跳过")
                                            continue
                                    if symmetric_flags[2]:
                                        mirror_move = board_inner.location_to_move((col, row))
                                        if moves.count(mirror_move) == 1:
                                            print("斜上轴上出现对称点，跳过")
                                            continue
                                    if symmetric_flags[3]:
                                        mirror_move = board_inner.location_to_move((board_inner.width - 1 - col, board_inner.height - 1 - row))
                                        if moves.count(mirror_move) == 1:
                                            print("斜下轴上出现对称点，跳过")
                                            continue
                                    temp_five_nums -= 1
                                    moves.append(move)
                                    x, y = board_inner.move_to_location(move)
                                    # print(board_inner.move_to_location(move), "        ", value)
                                    objects.append(
                                        GameObject(chess_list[2],
                                                   letter_list[0],
                                                   (27 + x * 40, 27 + (14 - y) * 40)))
                        print(board_inner.states)
                        hint_text = font.render("悟空已经完成打点，请你选择", True, word_color[0])  # 提示文案
                        screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
                        hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=200)  # 提示文案位置
                        screen.blit(hint_text, hint_text_pos)

                        for o in objects:
                            screen.blit(o.image, o.pos)
                        pg.display.update()

                        pg.time.delay(1000)  # 暂停1秒，保证文案能够清晰展示

                        flag = 0

                        while flag == 0:

                            screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
                            for o in objects:
                                screen.blit(o.image, o.pos)

                            pg.display.update()
                            for event in pg.event.get():
                                # 如果关闭窗口，主循环结束
                                if event.type == pg.QUIT:
                                    quit_game()
                                # 如果点击键盘ESC键，主循环结束
                                elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                                    quit_game()
                                # 如果玩家进行了鼠标点击操作
                                elif event.type == pg.MOUSEBUTTONDOWN:
                                    pos = pg.mouse.get_pos()  # 获取鼠标点击坐标

                                    # 若用户点击的不是悔棋、恢复按钮，则进行落子操作
                                    a, b = round((pos[0] - 27) / 40), round(
                                        (pos[1] - 27) / 40)  # 将用户鼠标点击位置的坐标，换算为board坐标
                                    # 若坐标非法（即点击到了黑色区域），则不做处理
                                    if a >= 15 or b >= 15:
                                        continue
                                    else:
                                        x, y = max(0, a) if a < 0 else min(a, 14), max(0, b) if b < 0 else min(b,
                                                                                                               14)  # 将a、b进行处理得到x和y
                                        print('*' * 50)
                                        print("棋盘1", x, 14 - y)
                                        print("棋盘2:", x, y)
                                        print("棋盘3：", chr(ord('A') + x), 15 - y)
                                        print('*' * 50)
                                        print()
                                        # 判断落子位置是否合法
                                        # 若落子操作合法，则进行落子

                                        if board_inner.location_to_move((x, 14 - y)) in moves:
                                            flag = 1
                                            for i in range(five_nums):
                                                objects.pop()
                                            board_inner.do_move(board_inner.location_to_move((x, 14 - y)))
                                            objects.append(
                                                GameObject(
                                                    chess_list[board_inner.current_player == board_inner.start_player],
                                                    letter_list[board_inner.current_player == board_inner.start_player],
                                                    (27 + x * 40, 27 + y * 40)))
                                            # 一旦成功落子，则将悔棋记录列表清空；不这么做，一旦在悔棋和恢复中间掺杂落子操作，就会有问题
                                            recover_objects.clear()

                                        else:
                                            hint_text = font.render("该位置落子不属于打点", True, word_color[0])  # 提示文案
                                            hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2,
                                                                               y=200)  # 提示文案位置
                                            # 将下棋记录的棋子打印到游戏窗口
                                            screen.blit(hint_text, hint_text_pos)  # 把提示文案打印到游戏窗口
                                            pg.display.update()  # 对游戏窗口进行刷新
                                            pg.time.delay(300)  # 暂停0.3秒，保证文案能够清晰展示
                                        for o in objects:
                                            screen.blit(o.image, o.pos)
                                        pg.display.update()  # 对游戏窗口进行刷新


                    else:
                        screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
                        print(f"该局面为{openings_dict[result][0]}，悟空不选择三手交换")
                        hint_text = font.render(f"该局面为{openings_dict[result][0]}，悟空不选择三手交换", True,
                                                word_color[1])  # 提示文案
                        hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=100)  # 提示文案位置
                        screen.blit(hint_text, hint_text_pos)
                        hint_text_temp = font.render("悟空已经完成落子，请你进行打点", True, word_color[1])  # 提示文案
                        hint_text_pos_temp = hint_text.get_rect(centerx=background.get_width() / 2, y=200)  # 提示文案位置
                        screen.blit(hint_text_temp, hint_text_pos_temp)
                        board_inner.do_move(openings_dict[result][1])
                        symmetric_flags = board_inner.is_symmetric()
                        for i in range(4):
                            if symmetric_flags[i]:
                                print("存在对称")


                        x, y = board_inner.move_to_location(openings_dict[result][1])
                        print("棋盘1", x, y)
                        print("棋盘2:", x, 14-y)
                        print("棋盘3：", chr(ord('A') + x), 1+ y)
                        print('*' * 50)
                        objects.append(
                            GameObject(chess_list[board_inner.current_player == board_inner.start_player],
                                       letter_list[board_inner.current_player == board_inner.start_player],
                                       (27 + x * 40, 27 + (14-y) * 40)))

                        # 停留1秒
                        for o in objects:
                            screen.blit(o.image, o.pos)

                        # x, y = board_inner.move_to_location(board_inner.last_move)
                        # temp = GameObject(chess_list[3],
                        #                   letter_list[0],
                        #                   (27 + x * 40, 27 + (14 - y) * 40))
                        # screen.blit(background, (0, 0))
                        # screen.blit(temp.image, temp.pos)
                        pg.display.update()
                        pg.time.delay(3000)  # 暂停1秒，保证文案能够清晰展示
                        cnt = 0
                        moves = []
                        while cnt < five_nums:
                            screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
                            text = font.render("第{}手".format(cnt + 1),
                                               True, word_color[
                                                   board_inner.current_player == board_inner.start_player])  # 创建一个文本对象，显示当前是哪方的回合
                            text_pos = text.get_rect(centerx=background.get_width() / 2, y=2)  # 确定文本对象的显示位置
                            screen.blit(text, text_pos)  # 将文本对象打印到游戏窗口
                            for o in objects:
                                screen.blit(o.image, o.pos)

                            pg.display.update()
                            for event in pg.event.get():
                                # 如果关闭窗口，主循环结束
                                if event.type == pg.QUIT:
                                    quit_game()
                                # 如果点击键盘ESC键，主循环结束
                                elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                                    quit_game()
                                # 如果玩家进行了鼠标点击操作
                                elif event.type == pg.MOUSEBUTTONDOWN:
                                    pos = pg.mouse.get_pos()  # 获取鼠标点击坐标

                                    # 若用户点击的不是悔棋、恢复按钮，则进行落子操作
                                    a, b = round((pos[0] - 27) / 40), round(
                                        (pos[1] - 27) / 40)  # 将用户鼠标点击位置的坐标，换算为board坐标
                                    # 若坐标非法（即点击到了黑色区域），则不做处理
                                    if a >= 15 or b >= 15:
                                        continue
                                    else:
                                        x, y = max(0, a) if a < 0 else min(a, 14), max(0, b) if b < 0 else min(b,
                                                                                                               14)  # 将a、b进行处理得到x和y
                                        print('*' * 50)
                                        print("棋盘1", x, 14 - y)
                                        print("棋盘2:", x, y)
                                        print("棋盘3：", chr(ord('A') + x), 15 - y)
                                        print('*' * 50)
                                        print()
                                        # 判断落子位置是否合法
                                        # 若落子操作合法，则进行落子
                                        move = board_inner.location_to_move((x, 14 - y))
                                       #TODO 这里需要判断是否为对称点
                                        row, col = board_inner.move_to_location(move)
                                        if symmetric_flags[0]:
                                            mirror_move = board_inner.location_to_move(
                                                (board_inner.height - 1 - row, col))
                                            if moves.count(mirror_move) == 1:
                                                print("出现水平轴上出现对称点，错误")
                                                continue
                                        if symmetric_flags[1]:
                                            mirror_move = board_inner.location_to_move(
                                                (row, board_inner.width - 1 - col))
                                            if moves.count(mirror_move) == 1:
                                                print("出现垂直轴上出现对称点，错误")
                                                continue
                                        if symmetric_flags[2]:
                                            mirror_move = board_inner.location_to_move((col, row))
                                            if moves.count(mirror_move) == 1:
                                                print("出现斜上轴上出现对称点，错误")
                                                continue
                                        if symmetric_flags[3]:
                                            mirror_move = board_inner.location_to_move(
                                                (board_inner.width - 1 - col, board_inner.height - 1 - row))
                                            if moves.count(mirror_move) == 1:
                                                print("出现斜下轴上出现对称点，错误")
                                                continue

                                        if move in board_inner.availables:
                                            moves.append(move)
                                            board_inner.do_move(move)
                                            objects.append(
                                                GameObject(chess_list[2],
                                                           letter_list[0],
                                                           (27 + x * 40, 27 + y * 40)))
                                            # 一旦成功落子，则将悔棋记录列表清空；不这么做，一旦在悔棋和恢复中间掺杂落子操作，就会有问题
                                            pro, value = AI_player.evaluate(board_inner)
                                            board_inner.undo_move(x, 14 - y)
                                            heapq.heappush(min_heap, (-value, move))
                                            cnt += 1

                                        else:
                                            hint_text = font.render("该位置已有棋子", True, word_color[0])  # 提示文案
                                            hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2,
                                                                               y=200)  # 提示文案位置
                                            # 将下棋记录的棋子打印到游戏窗口
                                        for o in objects:
                                            screen.blit(o.image, o.pos)
                                        screen.blit(hint_text, hint_text_pos)  # 把提示文案打印到游戏窗口

                                        pg.display.update()  # 对游戏窗口进行刷新
                                        pg.time.delay(300)  # 暂停0.3秒，保证文案能够清晰展示

                        for i in range(five_nums):
                            objects.pop()

                        # 选择最差点的落子
                        if min_heap:
                            value, move = heapq.heappop(min_heap)
                            value = -value
                            x, y = board_inner.move_to_location(move)
                            board_inner.do_move(move)
                            print(board_inner.move_to_location(move), "        ", value)
                            objects.append(
                                GameObject(chess_list[board_inner.current_player == board_inner.start_player],
                                           letter_list[board_inner.current_player == board_inner.start_player],
                                           (27 + x * 40, 27 + (14 - y) * 40)))
                            screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
                            for o in objects:
                                screen.blit(o.image, o.pos)

                            pg.display.update()

                else:
                    print("该开局不是指定开局，请你重新输入")
                    again_flag = 1

    temp_start_time=time.time()  # 记录开始时间
    while going:
        screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
        temp=1
        if board_inner.current_player == board_inner.start_player:
            temp=0
        text = font.render("{}回合".format(word_list[temp]), True, word_color[temp])  # 创建一个文本对象，显示当前是哪方的回合
        text_pos = text.get_rect(centerx=background.get_width() / 2+100, y=2)  # 确定文本对象的显示位置
        text_temp=font.render("{}回合".format(word_list_temp[board_inner.current_player == board_inner.AI_turn]), True, word_color[temp])  # 创建一个文本对象，显示当前是哪方的回合
        text_pos_temp = text_temp.get_rect(centerx=background.get_width() / 2-100, y=2)
        screen.blit(text_temp, text_pos_temp)
        screen.blit(text, text_pos)  # 将文本对象打印到游戏窗口
        # 显示AI总思考时间和人类总思考时间


        # 通过循环不断识别玩家操作
        for o in objects:
            screen.blit(o.image, o.pos)


        pg.display.update()  # 对游戏窗口进行刷新


        if board_inner.current_player == board_inner.AI_turn:
            # Action_flag = 0
            # if len(board_inner.states)==24:
            #     first_three_moves = list(board_inner.states.keys())[:24]
            #     # 将前三个move用空格连接成字符串
            #     result = ' '.join(map(str, first_three_moves))
            #     if result=="112 113 144 128 129 99 98 142 114 130 85 126 84 110 158 125 127 95 140 94 78 93 96 107":
            #              print(result)
            temp_start_time = time.time()  # 记录开始时间
            move = AI_player.get_action(board_inner)
            temp_end_time = time.time()
            cost_time = temp_end_time - temp_start_time
            cost_time_mins = cost_time // 60
            cost_time_secs = cost_time % 60
            temp_start_time=time.time()  # 记录开始时间
            print('*' * 50)
            print("悟空思考时间：", cost_time_mins, "分", cost_time_secs, "秒")
            AI_total_time += cost_time
            mins = AI_total_time // 60
            secs = AI_total_time % 60
            print("悟空总思考时间：", mins, "分", secs, "秒")
            print('*' * 50)

            x, y = board_inner.move_to_location(move)

            print("棋盘1：",x, y)
            print("棋盘2：",x,14-y)
            print()
            print("悟空落子：")
            print("棋盘3： ", chr(ord('A') + x),y+1)
            print('*' * 50)
            print()
            board_inner.do_move(move)
            objects.append(GameObject(chess_list[board_inner.current_player==board_inner.start_player], letter_list[board_inner.current_player==board_inner.start_player],
                                      (27 + x* 40, 27 + (14-y ) * 40)))

            # 判断是否出现获胜方
            end, winner = board_inner.game_end()
            if winner == board_inner.start_player:
                flag = 0
            else:
                flag = 1

            # 判断是否出现平局
            if end:
                # 将下棋记录的棋子打印到游戏窗口
                for o in objects:
                    screen.blit(o.image, o.pos)

                # 根据flag获取到当前获胜方，生成获胜文案
                if board_inner.have_non_compliance== 1:
                    reason="黑棋违规"
                else:
                    reason="成功连成五子"
                win_text = font.render(f"{word_list[flag]}获胜，原因：{reason}", True,
                                       word_color[flag])
                # 设定获胜文案的位置
                for o in objects:
                    screen.blit(o.image, o.pos)
                win_text_pos = win_text.get_rect(centerx=background.get_width() / 2, y=200)
                screen.blit(win_text, win_text_pos)  # 把获胜文案打印到游戏窗口
                pg.display.update()  # 对游戏窗口进行刷新

                win_flag = 0
                while not win_flag:
                    for event in pg.event.get():
                        if event.type == pg.QUIT:
                            quit_game()
                        elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                            quit_game()
                        elif event.type == pg.MOUSEBUTTONDOWN:
                            pos = pg.mouse.get_pos()
                            if continue_button.check_click(pos):
                                win_flag = 1
                                current_time = datetime.now().strftime('%Y-%m-%d %H:%M')

                                if start_player==0:
                                    beginner = "悟空五子棋"
                                    Next_player = "敌方队伍"
                                else:
                                    beginner = "敌方队伍"
                                    Next_player = "悟空五子棋"

                                if board_inner.start_player == board_inner.AI_turn and start_player == 0:
                                    winner = "先手胜" if flag == 0 else "后手胜"
                                elif board_inner.start_player != board_inner.AI_turn and start_player == 0:
                                    winner = "先手胜" if flag == 1 else "后手胜"
                                elif board_inner.start_player == board_inner.AI_turn and start_player == 1:
                                    winner = "先手胜" if flag == 1 else "后手胜"
                                elif board_inner.start_player != board_inner.AI_turn and start_player == 1:
                                    winner = "先手胜" if flag == 0 else "后手胜"

                                date = current_time
                                location = "东北大学秦皇岛分校"
                                event = "2025 CCGC"

                                return_Str = f"{{[C5][{beginner} ][{Next_player} ][{winner}][{date} {location}][{event}]"

                                cnt = 0
                                for move in board_inner.states.keys():
                                    temp_str = ";"
                                    now_str = "B" if cnt % 2 == 0 else "W"
                                    temp_str += now_str
                                    temp_str += "("
                                    x, y = board_inner.move_to_location(move)
                                    temp_str += chr(ord('A') + x)
                                    temp_str += ","
                                    temp_str += str(y + 1)
                                    temp_str += ")"
                                    cnt += 1
                                    return_Str += temp_str
                                return_Str += "}"
                                AI_tot_mins = AI_total_time // 60
                                AI_tot_secs = AI_total_time % 60
                                print("AI 总思考时间：", AI_tot_mins, "分", AI_tot_secs, "秒")
                                print("对手总思考时间：", human_total_time // 60, "分", human_total_time % 60, "秒")
                                print("总下棋次数：", board_inner.states.__len__())
                                total_time = time.time() - begin_time
                                total_mins = total_time // 60
                                total_secs = total_time % 60
                                print("总下棋时间：", total_mins, "分", total_secs, "秒")
                                print(return_Str)
                                # 保存棋谱到文件
                                now_time = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
                                filename = f"棋谱{now_time}.txt"
                                filename = os.path.join("棋谱", filename)
                                # 检查文件是否存在
                                if os.path.exists(filename):
                                    mode = 'a'  # 文件存在，追加写入
                                else:
                                    mode = 'w'  # 文件不存在，创建并写入

                                with open(filename, mode, encoding="utf-8") as f:
                                    f.write(return_Str)
                                going=False



                            elif regret_button.check_click(pos):
                                for i in range(2):
                                    x, y = [round((p + 18 - 27) / 40) for p in ob_list[0][-1].pos[:2]]
                                    board_inner.undo_move(x, 14 - y)
                                    ob_list[- 1].append(
                                        ob_list[0][-1])  # 将游戏/悔棋记录列表的最后一个值添加到悔棋/下棋记录列表
                                    ob_list[0].pop()  # 将游戏/悔棋记录列表的最后一个值删除
                                    win_flag = 1
        # 若落子位置已经有棋子，则进行提示


        else:
            for event in pg.event.get():
                # 如果关闭窗口，主循环结束
                if event.type == pg.QUIT:
                    quit_game()
                # 如果点击键盘ESC键，主循环结束
                elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    quit_game()
                # 如果玩家进行了鼠标点击操作
                elif event.type == pg.MOUSEBUTTONDOWN:
                    pos = pg.mouse.get_pos()  # 获取鼠标点击坐标

                    # 如果点击了悔棋按钮或者恢复按钮
                    if regret_button.check_click(pos) or recover_button.check_click(pos):
                        index = 0 if regret_button.check_click(pos) else 1  # 点击悔棋按钮index = 0，点击恢复按钮index = 1
                        # 对指定列表进行判空操作，然后对下棋记录列表或者悔棋记录列表进行操作

                        if ob_list[index]:
                            # print(ob_list[index][-1].pos)
                            # 将游戏/悔棋记录列表里的图像坐标，转化为board坐标
                            # 人机对战需要黑方、白方各悔一步棋；（如果只是玩家悔棋，AI会立即下出一步，导致悔棋失败）
                            for i in range(2):
                                x, y = [round((p + 18 - 27) / 40) for p in ob_list[index][-1].pos[:2]]

                                print('*' * 50)
                                print("棋盘1：", x, 14-y)
                                print("棋盘2：", x, y)
                                print("棋盘3：", chr(ord('A') + x), 15 - y)
                                print('*' * 50)
                                print()
                                # 如果是悔棋操作，则board指定元素值恢复为' '；如果是恢复操作，则指定坐标board指定元素重新赋值
                                if index == 0:
                                    board_inner.undo_move(x,14-y)
                                else :
                                    board_inner.do_move(board_inner.location_to_move((x,14-y)))

                                ob_list[index - 1].append(ob_list[index][-1])  # 将游戏/悔棋记录列表的最后一个值添加到悔棋/下棋记录列表
                                ob_list[index].pop()  # 将游戏/悔棋记录列表的最后一个值删除

                    else:
                        print("请你落子")
                        # 若用户点击的不是悔棋、恢复按钮，则进行落子操作
                        a, b = round((pos[0] - 27) / 40), round((pos[1] - 27) / 40)  # 将用户鼠标点击位置的坐标，换算为board坐标
                        # 若坐标非法（即点击到了黑色区域），则不做处理
                        if a >= 15 or b >= 15:
                            continue
                        else:
                            x, y = max(0, a) if a < 0 else min(a, 14), max(0, b) if b < 0 else min(b, 14)  # 将a、b进行处理得到x和y
                            print('*' * 50)

                            print("棋盘1",x, 14 - y)
                            print("棋盘2:",x,y)

                            print("棋盘3：", chr(ord('A') + x), 15 - y)
                            print('*' * 50)

                            print()
                            # 判断落子位置是否合法
                            # 若落子操作合法，则进行落子

                            if board_inner.location_to_move((x, 14 - y)) not in board_inner.states.keys():
                                board_inner.do_move(board_inner.location_to_move((x, 14 - y)))
                                objects.append(
                                    GameObject(chess_list[board_inner.current_player == board_inner.start_player],
                                               letter_list[board_inner.current_player == board_inner.start_player],
                                               (27 + x * 40, 27 + y * 40)))
                                # 一旦成功落子，则将悔棋记录列表清空；不这么做，一旦在悔棋和恢复中间掺杂落子操作，就会有问题
                                recover_objects.clear()
                                temp_end_time = time.time()
                                cost_time = temp_end_time - temp_start_time
                                cost_time_mins = cost_time // 60
                                cost_time_secs = cost_time % 60
                                print('*' * 50)
                                print("对手思考时间：", cost_time_mins, "分", cost_time_secs, "秒")
                                human_total_time += cost_time
                                mins = human_total_time // 60
                                secs = human_total_time % 60
                                print("对手总思考时间：", mins, "分", secs, "秒")
                                print('*' * 50)


                                # 判断是否出现获胜方
                                end,winner=board_inner.game_end()
                                if winner==board_inner.start_player:
                                    flag=0
                                else:
                                    flag=1


                                # 判断是否出现平局
                                if end:
                                    # 将下棋记录的棋子打印到游戏窗口
                                    for o in objects:
                                        screen.blit(o.image, o.pos)



                                    if board_inner.have_non_compliance == 1:
                                        reason = "黑棋违规"
                                    else:
                                        reason = "成功连成五子"
                                    win_text = font.render(f"{word_list[flag]}获胜，原因：{reason}", True,
                                                           word_color[flag])
                                    # 设定获胜文案的位置
                                    win_text_pos = win_text.get_rect(centerx=background.get_width() / 2, y=200)
                                    # 将获胜文案打印到游戏窗口
                                    for o in objects:
                                        screen.blit(o.image, o.pos)
                                    pg.display.update()  # 对游戏窗口进行刷新
                                    screen.blit(win_text, win_text_pos)  # 把获胜文案打印到游戏窗口
                                    pg.display.update()  # 对游戏窗口进行刷新

                                    win_flag=0
                                    while not win_flag:
                                        for event in pg.event.get():
                                            if event.type == pg.QUIT:
                                                quit_game()
                                            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                                                quit_game()
                                            elif event.type == pg.MOUSEBUTTONDOWN:
                                                pos = pg.mouse.get_pos()
                                                if continue_button.check_click(pos):
                                                    win_flag = 1
                                                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M')

                                                    if start_player == 0:
                                                        beginner = "悟空五子棋"
                                                        Next_player = "敌方队伍"
                                                    else:
                                                        beginner = "敌方队伍"
                                                        Next_player = "悟空五子棋"

                                                    if board_inner.start_player == board_inner.AI_turn and start_player == 0:
                                                        winner = "先手胜" if flag == 0 else "后手胜"
                                                    elif board_inner.start_player != board_inner.AI_turn and start_player == 0:
                                                        winner = "先手胜" if flag == 1 else "后手胜"
                                                    elif board_inner.start_player == board_inner.AI_turn and start_player == 1:
                                                        winner = "先手胜" if flag == 1 else "后手胜"
                                                    elif board_inner.start_player != board_inner.AI_turn and start_player == 1:
                                                        winner = "先手胜" if flag == 0 else "后手胜"

                                                    date = current_time
                                                    location = "东北大学秦皇岛分校"
                                                    event = "2025 CCGC"

                                                    return_Str = f"{{[C5][{beginner} ][{Next_player} ][{winner}][{date} {location}][{event}]"

                                                    cnt = 0
                                                    for move in board_inner.states.keys():
                                                        temp_str =";"
                                                        now_str="B" if cnt%2==0 else "W"
                                                        temp_str+=now_str
                                                        temp_str+="("
                                                        x,y=board_inner.move_to_location(move)
                                                        temp_str+=chr(ord('A')+x)
                                                        temp_str += ","
                                                        temp_str+=str(y+1)
                                                        temp_str+=")"
                                                        cnt+=1
                                                        return_Str+=temp_str
                                                    return_Str+="}"

                                                    AI_tot_mins = AI_total_time // 60
                                                    AI_tot_secs = AI_total_time % 60
                                                    print("AI 总思考时间：", AI_tot_mins, "分", AI_tot_secs, "秒")
                                                    print("对手总思考时间：", human_total_time // 60, "分",human_total_time % 60, "秒")
                                                    print("总下棋次数：", board_inner.states.__len__())
                                                    total_time = time.time() - begin_time
                                                    total_mins = total_time // 60
                                                    total_secs = total_time % 60
                                                    print("总下棋时间：", total_mins, "分", total_secs, "秒")
                                                    print(return_Str)
                                                    now_time = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
                                                    filename = f"棋谱{now_time}.txt"
                                                    filename = os.path.join("棋谱", filename)
                                                    # 检查文件是否存在
                                                    if os.path.exists(filename):
                                                        mode = 'a'  # 文件存在，追加写入
                                                    else:
                                                        mode = 'w'  # 文件不存在，创建并写入

                                                    with open(filename, mode, encoding="utf-8") as f:
                                                        f.write(return_Str)
                                                    quit_game()


                                                # 保存棋谱到数据库



                                                elif regret_button.check_click(pos):
                                                    for i in range(2):
                                                        x, y = [round((p + 18 - 27) / 40) for p in
                                                                ob_list[0][-1].pos[:2]]
                                                        board_inner.undo_move(x, 14 - y)
                                                        ob_list[- 1].append(
                                                        ob_list[0][-1])  # 将游戏/悔棋记录列表的最后一个值添加到悔棋/下棋记录列表
                                                        ob_list[0].pop()  # 将游戏/悔棋记录列表的最后一个值删除
                                                        win_flag=1

                            else:
                                hint_text = font.render("该位置已有棋子", True, word_color[0])  # 提示文案
                                hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=200)  # 提示文案位置
                                # 将下棋记录的棋子打印到游戏窗口
                                for o in objects:
                                    screen.blit(o.image, o.pos)
                                screen.blit(hint_text, hint_text_pos)  # 把提示文案打印到游戏窗口

                                pg.display.update()  # 对游戏窗口进行刷新
                                pg.time.delay(300)  # 暂停0.3秒，保证文案能够清晰展示
            # AI执黑，AI进行落子




        # 将下棋记录的棋子打印到游戏窗口
        for o in objects:
            screen.blit(o.image, o.pos)

        clock.tick(60)  # 游戏帧率每秒60帧
        pg.display.update()  # 对游戏窗口进行刷新




if __name__ == '__main__':
    externalProgramManager = gomoku_engine.Board()
    hash_table_manager = game.HashTableManager("merged_hash_new2.pkl")
    board = game.Board(ExternalProgramManager=externalProgramManager,hash_table_manager=hash_table_manager,width=15, height=15, n_in_row=5)
    #Best_player = players.AIplayer(c_puct=2, n_playout=600, is_selfplay=False)
    # now_temp_player = players.AIPlayer_MCTS(model_path="best_model/current_policy.model4200",c_puct=2, n_playout=200, is_selfplay=False)
    Best_player = players.AIplayer("best_model/current_policy_step_best.model")
    for i in range(2):
        board.init_board()
        pg.init()
        main_dir = os.path.split(os.path.abspath(__file__))[0]
        font = pg.font.Font('font/12345.TTF', 20)
        font_big = pg.font.Font('font/12345.TTF', 40)
        main(board, Best_player)
        pg.quit()
    externalProgramManager.terminate()
