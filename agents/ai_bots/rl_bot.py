import random
import numpy as np
from agents.base_agent import BaseAgent

class RLBot(BaseAgent):
    def __init__(self, name="RLBot", player_id=1, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        """
        简化版强化学习Bot
        
        参数:
            name: 智能体名称
            player_id: 玩家ID (1或2)
            learning_rate: 学习率 (alpha)
            discount_factor: 折扣因子 (gamma)
            exploration_rate: 探索率 (epsilon)
        """
        super().__init__(name, player_id)
        self.q_table = {}  # 简化的Q表
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.last_state = None
        self.last_action = None
        
    def _state_to_key(self, state):
        """将游戏状态转换为Q表的键(简化版)"""
        return str(state)  # 实际项目应该设计更好的状态表示
        
    def get_action(self, observation, env):
        """
        根据当前状态选择动作
        
        参数:
            observation: 当前游戏状态
            env: 游戏环境
        
        返回:
            action: 选择的动作
        """
        valid_actions = env.get_valid_actions(self.player_id)
        if not valid_actions:
            return None
            
        state_key = self._state_to_key(observation)
        
        # 初始化Q表条目(如果不存在)
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0 for action in valid_actions}
        
        # ε-贪婪策略
        if random.random() < self.exploration_rate:
            # 探索: 随机选择动作
            action = random.choice(valid_actions)
        else:
            # 利用: 选择Q值最高的动作
            q_values = self.q_table[state_key]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = random.choice(best_actions)  # 如果有多个相同Q值则随机选
            
        # 保存当前状态和动作用于学习
        self.last_state = state_key
        self.last_action = action
        
        return action
        
    def learn(self, state, action, reward, next_state, done):
        """
        更新Q表 (简化版Q学习)
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        if self.last_state is None:
            return
            
        current_q = self.q_table[self.last_state].get(self.last_action, 0)
        
        # 计算目标Q值
        if done:
            target = reward
        else:
            next_state_key = self._state_to_key(next_state)
            max_next_q = max(self.q_table.get(next_state_key, {}).values(), default=0)
            target = reward + self.discount_factor * max_next_q
        
        # Q表更新
        self.q_table[self.last_state][self.last_action] = (1 - self.learning_rate) * current_q + self.learning_rate * target
        
    def update_exploration(self, episode, total_episodes):
        """随时间减少探索率(可选)"""
        self.exploration_rate = max(0.01, 0.3 * (1 - episode / total_episodes))
        
    def save_model(self, filepath):
        """保存模型(简化版)"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
            
    def load_model(self, filepath):
        """加载模型(简化版)"""
        import pickle
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
