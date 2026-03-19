from src.models.policy_value_net import PolicyValueNet
from src.models.policy_value_utss_net import PolicyValueUTSSNet
from src.game import Game, Board,HashTableManager
from src.players import AIplayer,Human
from src.mcts.mcts_alphazero1 import MCTSPlayer as MCTSPlayer1
from src.mcts.mcts_alphazero2 import MCTSPlayer as MCTSPlayer2
from src.models.tss_classifier_old import TSSClassifier
import torch
import gomoku_engine

from trueskill import Rating, quality_1vs1, rate_1vs1
import random
from collections import defaultdict

if __name__ =="__main__":
    externalProgramManager = gomoku_engine.Board()
    hash_table_manager = HashTableManager("merged_hash_new2.pkl")
    board = Board(ExternalProgramManager=externalProgramManager,hash_table_manager=hash_table_manager,width=15, height=15, n_in_row=5)
    board.init_board()
    game = Game(board,True)
    tss_classifier = TSSClassifier(num_channels=128, num_res_blocks=7).to('cuda' if torch.cuda.is_available() else 'cpu')
    tss_classifier.load_state_dict(torch.load('best_model/tss_classifier_800.pth'))
    tss_classifier.eval()

     # 初始化模型和它们的TrueSkill评分
    models = {f"Model_{i*50}": Rating() for i in range(16)}
    wins = defaultdict(int)
    games_played = defaultdict(int)

    # 加载模型
    model_players = []
    for i in range(0,17):
        model_file = f'model\\current_threeHead.model{i*50}'
        try:
            policy_value_utss_net = PolicyValueUTSSNet(15, 15, model_file=model_file)
            player = MCTSPlayer2(policy_value_utss_net.policy_value_utss_fn, 5, 100, 0)
            model_players.append(player)
            print(f"成功加载模型：{model_file}")
        except FileNotFoundError:   
            print(f"警告：无法找到模型文件 {model_file}，跳过此模型")
            continue

    model_file = 'best_model\\current_policy_step_best.model'
    try:
        aiplayer = AIplayer(model_path = model_file)
        mcts_player_best = aiplayer.mcts_player
        model_players.append(mcts_player_best)
        print(f"成功加载模型：{model_file}")
    except FileNotFoundError:
        print(f"警告：无法找到模型文件 {model_file}，跳过此模型")

    # 确保至少有两个模型被加载
    if len(model_players) < 2:
        raise ValueError("至少需要两个有效的模型才能进行比较")

     # 对弈和更新TrueSkill
    num_games = 100  # 每对模型之间的对弈次数，减半
    for _ in range(num_games):
        for i in range(len(model_players)):
            for j in range(i+1, len(model_players)):
                player1 = model_players[i]
                player2 = model_players[j]
                
                # 计算对局质量
                quality = quality_1vs1(models[f"Model_{i*50}"], models[f"Model_{j*50}"])
                
                for start_player in [0, 1]:  # 0 表示 player1 先手，1 表示 player2 先手
                    # 进行对弈
                    try:
                        winner = game.start_play(player1, player2, i, j, start_player=start_player)
                        
                        # 更新TrueSkill评分
                        if winner == 0 :  # player1 赢
                            models[f"Model_{i*50}"], models[f"Model_{j*50}"] = rate_1vs1(models[f"Model_{i*50}"], models[f"Model_{j*50}"])
                            wins[f"Model_{i*50}"] += 1
                        else:  # player2 赢
                            models[f"Model_{j*50}"], models[f"Model_{i*50}"] = rate_1vs1(models[f"Model_{j*50}"], models[f"Model_{i*50}"])
                            wins[f"Model_{j*50}"] += 1
                        
                        games_played[f"Model_{i*50}"] += 1
                        games_played[f"Model_{j*50}"] += 1

                        print(f"Game {_*240 + i*30 + j*2 + start_player}: Model_{i*50} vs Model_{j*50}, Start Player: Model_{start_player}, Winner: Model_{winner}, Quality: {quality:.2%}")

                    except Exception as e:
                        print(f"在对弈 Model_{i*50} vs Model_{j*50} (Start Player: Model_{start_player}) 时发生错误: {str(e)}")


    # 打印最终TrueSkill评分和胜率
    for model, rating in sorted(models.items(), key=lambda x: x[1].mu, reverse=True):
        win_rate = wins[model] / games_played[model] if games_played[model] > 0 else 0
        print(f"{model}: TrueSkill = {rating.mu:.2f} ± {rating.sigma:.2f}, Win rate = {win_rate:.2%}")
        # 将结果保存到文件中
        with open('result\\elo_rate_result.txt', 'a') as f:
            f.write(f"{model}: TrueSkill = {rating.mu:.2f} ± {rating.sigma:.2f}, Win rate = {win_rate:.2%}\n")


