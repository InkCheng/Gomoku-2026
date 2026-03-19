import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import pandas as pd
from pathlib import Path

LOG_PATH = Path("train_log.csv")

class ResBlock(nn.Module):

    def __init__(self, num_filters=128):
        super().__init__()
        # 使用3x3的卷积核，步长为1，填充为1
        self.conv1 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


# 三头网络结构类，输入：N, 9, 10, 9 --> N, C, H, W
class Net(nn.Module):
    def __init__(self, in_channels=7, num_channels=128, num_res_blocks=7, board_size=15):
        super().__init__()
        self.board_size = board_size

        self.policy_bottle = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )
        self.value_bottle = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )
        self.utss_bottle = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )

        # 初始化特征(公共卷积主干)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)  # 正则项
        )
        self.res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])  # 残差块抽取特征

        # 策略头
        self.policy_conv = nn.Conv2d(num_channels, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
        self.policy_bn = nn.BatchNorm2d(16)
        # self.policy_act = nn.ReLU()
        self.policy_fc = nn.Linear(16 * board_size * board_size, board_size * board_size)

        # 价值头
        self.value_conv = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * board_size * board_size, 225)
        self.value_fc2 = nn.Linear(225, 1)

        # TSS头
        self.utss_conv = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
        self.utss_bn = nn.BatchNorm2d(16)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.utss_fc = nn.Sequential(
            nn.Linear(16,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )

    # 定义前向传播
    def forward(self, x, use_tss = True):
        x = x.reshape(-1, 7, 15, 15)
        # 公共头
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
        feat = x
        p_feat = self.policy_bottle(feat)
        v_feat = self.value_bottle(feat)
        utss_feat = self.utss_bottle(feat)
        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(p_feat)), inplace=True)
        policy = policy.view(policy.size(0), -1)
        log_act_probs = F.log_softmax(self.policy_fc(policy), dim=1)

        # 价值头
        value = F.relu(self.value_bn(self.value_conv(v_feat)), inplace=True)
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value), inplace=True)
        value = torch.tanh(self.value_fc2(value))

        # UTSS头
        if use_tss:
            utss = self.utss_conv(utss_feat)
            utss = self.utss_bn(utss)
            utss = self.global_avg_pool(utss)
            utss = utss.view(utss.size(0), -1)
            utss_logits = self.utss_fc(utss)
        else:
            utss_logits = None

        return log_act_probs, value, utss_logits

class PolicyValueUTSSNet():
    """策略 + 价值 + UTSS 网络封装"""
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.device = torch.device("cuda") if self.use_gpu else torch.device("cpu")
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-3  # coef of l2 penalty
        self.value_weight = 0.067
        self.gamma_vtss = 0.2
        self.sa_scheduler = LambdaPIDController()

        self.policy_value_utss_net = Net(7,128,7,15).to(self.device)

        self.optimizer = torch.optim.Adam(lr=0.002,
                                          params=self.policy_value_utss_net.parameters(),
                                          weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_utss_net.load_state_dict(net_params)

    def collect_utss_training_data(self, state, result):
        """
        收集 UTSS 标签用于训练
        result:
        - 0: Win
        - 1: Draw
        - 2: Loss
        - 3: Unknown
        """
        one_hot_result = [0, 0, 0, 0]
        if 0 <= result <= 3:
            one_hot_result[result] = 1

        if not hasattr(self, 'utss_training_data'):
            self.utss_training_data = []

        self.utss_training_data.append((state, one_hot_result))
        if len(self.utss_training_data) > 10000:
            self.utss_training_data.pop(0)

    def policy_value_utss(self, state_batch, use_tss=True):
        """
       input: a batch of states
       output: a batch of action probabilities and state values
       """
        self.policy_value_utss_net.eval()
        state_batch = torch.as_tensor(dtype=torch.float32, data=state_batch).to(self.device)
        log_act_probs, values, utss_logits = self.policy_value_utss_net(state_batch, use_tss)
        act_probs = np.exp(log_act_probs.detach().cpu().numpy())
        return act_probs, values.detach().cpu().numpy(), utss_logits


    def policy_value_utss_fn(self, board, use_tss=True):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        # 数组在内存中存放的地址也是连续的
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 7, self.board_width, self.board_height)).astype("float32")
        act_probs, value, utss_logits = self.policy_value_utss(current_state, use_tss)
        act_probs = act_probs.flatten()
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return list(act_probs), value, utss_logits

    def train_step(self, state_batch, mcts_probs, winner_batch, utss_batch=None, lr=0.002):
        """perform a training step(三头联合训练)"""
        self.policy_value_utss_net.train()

        state_batch = torch.from_numpy(state_batch).to(self.device)
        mcts_probs = torch.from_numpy(mcts_probs).to(self.device)
        winner_batch = torch.from_numpy(winner_batch).to(self.device)

        if utss_batch is not None:
            utss_batch = torch.from_numpy(utss_batch).to(self.device).long()

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        for params in self.optimizer.param_groups:
            params['lr'] = lr

        # forward
        log_act_probs, value, utss_logits = self.policy_value_utss_net(state_batch, use_tss=(utss_batch is not None))
        value = value.view(-1)

        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))
        value_loss = F.mse_loss(input=value, target=winner_batch)

        if utss_batch is not None and utss_logits is not None:
            # 计算vtss(价值头与tss头约束)
            p_win = F.softmax(utss_logits, dim=1)[:, 0]  # “必胜” 类
            p_loss = F.softmax(utss_logits, dim=1)[:, 2]  # “必败” 类
            v_hat = (value + 1) / 2  # value ∈ [-1,1] → [0,1]
            v_hat = torch.clamp(v_hat, 1e-6, 1 - 1e-6)
            vtss_loss = F.binary_cross_entropy(input=v_hat, target=p_win) + F.binary_cross_entropy(input=1 - v_hat, target=p_loss)

            utss_loss = F.cross_entropy(utss_logits, utss_batch)

            # 计算utss_weight
            pred_labels = torch.argmax(utss_logits, dim=1).cpu().numpy()
            true_labels = utss_batch.cpu().numpy()
            utss_f1 = f1_score(true_labels, pred_labels, average='macro')
            utss_weight = self.sa_scheduler.update(utss_f1)

            total_loss = self.value_weight * value_loss + policy_loss + utss_weight * utss_loss
            total_loss += self.gamma_vtss * vtss_loss  # 增加价值头与tss头约束

            utss_loss_val = utss_loss.item()
        else:
            total_loss = self.value_weight * value_loss + policy_loss
            utss_loss_val = 0.0  # 不参与损失时，显示为0

        # backward and optimize
        total_loss.backward()
        self.optimizer.step()

        # 计算策略的熵，仅用于评估模型
        with torch.no_grad():
            entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1))

        print(f"p_loss: {policy_loss.item():.4f}, v_loss: {value_loss.item():.4f}, u_loss: {utss_loss_val:.4f}, vtss_loss: {vtss_loss.item():.4f}")
        return policy_loss.item(), value_loss.item(), utss_loss_val, vtss_loss.item(), total_loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_utss_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)

class LambdaPIDController:
    def __init__(self,
                 kp=0.3, ki=0.01, kd=0.05,
                 lam_min=0.05, lam_max=1.0,
                 target_f1=0.80,
                 log_path="lambda_pid_monitor_log.csv"):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.target_f1 = target_f1
        self.integral = 0.0
        self.prev_error = 0.0
        self.current_lambda = 0.3

        self.log_path = log_path
        if log_path:
            pd.DataFrame(columns=[
                'step', 'f1_score', 'error', 'lambda', 'P', 'I', 'D'
            ]).to_csv(log_path, index=False)

        self.step_counter = 0

    def update(self, current_f1):
        error = self.target_f1 - current_f1
        self.integral += error
        derivative = error - self.prev_error

        P = self.kp * error
        I = self.ki * self.integral
        D = self.kd * derivative

        delta = P + I + D
        self.current_lambda += delta
        self.current_lambda = np.clip(self.current_lambda, self.lam_min, self.lam_max)

        self.prev_error = error
        self.step_counter += 1

        if self.log_path:
            pd.DataFrame([{
                'step': self.step_counter,
                'f1_score': current_f1,
                'error': error,
                'lambda': self.current_lambda,
                'P': P,
                'I': I,
                'D': D
            }]).to_csv(self.log_path, mode='a', index=False, header=False)

        return self.current_lambda


# 保存日志
def append_log(f1_score,                # f1-score
               lam_value,               # value_weight
               lam_utss):               # 当前 λ (utss_weight)
    row = {
        "f1_score": f1_score,
        "lam_value": lam_value,
        "lambda_utss": lam_utss,
    }
    # DataFrame 一行写盘，mode='a' 表示 append
    pd.DataFrame([row]).to_csv(
        LOG_PATH,
        mode="a",
        header=not LOG_PATH.exists(),   # 首次写入才带表头
        index=False,
        float_format="%.6g"             # 防止科学计数写成 0.000000
    )
