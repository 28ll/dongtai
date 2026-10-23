import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class DQNScheduler:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=0.9, device='cpu'):
        """
        初始化 DQN 调度器

        参数:
            state_dim (int): 状态向量的维度（应与 get_state 返回的实际长度一致）
            action_dim (int): 动作空间的大小（应与实际可选消防站数量一致）
        """
        self.device = torch.device(device)
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim
        self.state_dim = state_dim

        # 确保推理时关闭 Dropout（也可以在 select_action 中临时设置）
        self.q_network.eval()
        self.target_network.eval()

    def get_state(self, fire_level, stations_status, traffic_state, weather_encoding):
        """
        根据消防站状态、交通、天气等信息构建状态向量

        注意：状态向量的长度由本函数返回的实际长度决定，需与初始化时的 state_dim 一致。
        如果长度不足，会自动补零；超出则截断。
        """
        state = [fire_level / 3.0]
        for s in stations_status:
            state.extend([
                s.get('distance', 0) / 5000.0,
                s.get('idle_count', 0) / 13.0,
                s.get('total_capacity', 0) / 200.0,
                s.get('avg_fill_time', 10) / 15.0,
                s.get('heavy_ratio', 0)
            ])
        state.extend(traffic_state)
        state.append(weather_encoding)

        # 对齐到指定维度
        if len(state) < self.state_dim:
            state.extend([0] * (self.state_dim - len(state)))
        state = state[:self.state_dim]
        return np.array(state, dtype=np.float32)

    def select_action(self, state, available_actions_mask):
        """
        根据 epsilon-greedy 策略选择动作

        参数:
            state (np.ndarray): 状态向量
            available_actions_mask (list): 长度为 action_dim 的布尔/0-1 列表，指示哪些动作可用
        """
        # 检查掩码长度是否与动作维度匹配，若不匹配则截断或补全
        if len(available_actions_mask) != self.action_dim:
            # 如果掩码长度小于 action_dim，用 0 补齐；大于则截断
            mask = [0] * self.action_dim
            for i in range(min(len(available_actions_mask), self.action_dim)):
                mask[i] = available_actions_mask[i]
        else:
            mask = list(available_actions_mask)

        # 随机探索
        if random.random() < self.epsilon:
            valid = [i for i, m in enumerate(mask) if m]
            if not valid:
                # 若没有可用动作，返回 0（后续逻辑会处理）
                return 0
            return random.choice(valid)
        else:
            # 利用：选择 Q 值最大的可用动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy()[0]
            # 将不可用动作的 Q 值设为负无穷，避免被选中
            masked_q = np.where(np.array(mask, dtype=bool), q_values, -np.inf)
            return int(np.argmax(masked_q))

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, batch_size=64):
        """执行一步训练，返回损失值"""
        if len(self.memory) < batch_size:
            return 0.0
        # 训练时切换到训练模式
        self.q_network.train()
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 训练后恢复 eval 模式，避免影响后续推理
        self.q_network.eval()
        return loss.item()

    def update_target_network(self):
        """将目标网络的权重更新为当前 Q 网络的权重"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def save_model(self, path):
        """保存模型权重"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        """加载模型权重"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.q_network.eval()
        self.target_network.eval()