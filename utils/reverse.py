openings_dict = {
    # 直指开局(左右对称)
    "112 113 114": ("寒星局", 6 * 15 + 6, 8),
    "112 113 129": ("溪月局", 9 * 15 + 7, 18),
    "112 113 144": ("疏星局", 8 * 15 + 8, 3),
    "112 113 128": ("花月局", 6 * 15 + 6, 12),
    "112 113 143": ("残月局", 9 * 15 + 7, 20),
    "112 113 127": ("雨月局", 9 * 15 + 7, 10),
    "112 113 142": ("金星局", 8 * 15 + 9, 14),
    "112 113 111": ("松月局", 7 * 15 + 5, 17),
    "112 113 126": ("丘月局", 6 * 15 + 8, 5),
    "112 113 141": ("新月局", 6 * 15 + 8, 6),
    "112 113 110": ("瑞星局", 8 * 15 + 7, 9),
    "112 113 125": ("山月局", 5 * 15 + 6, 12),
    "112 113 140": ("游星局", 6 * 15 + 8, 0),

    # 斜指开局(对角线对称)
    "112 128 144": ("长星局", 7 * 15 + 9, 1),
    "112 128 143": ("峡月局", 9 * 15 + 6, 17),
    "112 128 142": ("恒星局", 8 * 15 + 7, 5),
    "112 128 141": ("水月局", 6 * 15 + 7, 16),
    "112 128 140": ("流星局", 6 * 15 + 8, 0),
    "112 128 127": ("云月局", 6 * 15 + 7, 9),
    "112 128 126": ("浦月局", 6 * 15 + 8, 9),
    "112 128 125": ("岚月局", 6 * 15 + 7, 11),
    "112 128 111": ("银月局", 7 * 15 + 5, 16),
    "112 128 110": ("明星局", 7 * 15 + 8, 9),
    "112 128 96": ("斜月局", 8 * 15 + 7, 2),
    "112 128 95": ("名月局", 8 * 15 + 7, 5),
    "112 128 80": ("彗星局", 9 * 15 + 8, 0),




}



def mirror_move(move, board_size=15):
    row, col = divmod(move, board_size)
    return (board_size - 1 - row) * board_size + col
    # return row * board_size + (board_size - 1 - col)

def diagonal_mirror_move(move, board_size=15):
    row, col = divmod(move, board_size)
    return col * board_size + row

# def mirror_position(pos, board_size=15):
#     return board_size - 1 - pos

def create_symmetric_openings(openings_dict):
    new_dict = {}
    for key, value in openings_dict.items():
        moves = list(map(int, key.split()))
        name, best_move, count = value

        # 添加原始局面
        new_dict[key] = value

        # 对于直指开局（前13个）
        if moves[1] == 113:
            mirrored_moves = [mirror_move(move) for move in moves]
            if mirrored_moves != moves:  # 如果不在对称轴上
                new_key = " ".join(map(str, mirrored_moves))
                new_best_move = mirror_move(best_move)
                new_dict[new_key] = (name, new_best_move, count)

        # 对于斜指开局（后13个）
        else:
            mirrored_moves = [diagonal_mirror_move(move) for move in moves]
            if mirrored_moves != moves:  # 如果不在对称轴上
                new_key = " ".join(map(str, mirrored_moves))
                new_best_move=diagonal_mirror_move(best_move)
                # row, col = divmod(best_move, 15)
                # new_best_move = col * 15 + row
                new_dict[new_key] = (name, new_best_move, count)

    return new_dict

# 创建新的包含对称局面的字典
symmetric_openings = create_symmetric_openings(openings_dict)

# 打印结果
for key, value in symmetric_openings.items():
    print(f"\" {key}\" : {value},")
