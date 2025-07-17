"""
五子棋游戏逻辑 - 修复版本
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from games.base_game import BaseGame
import config


class GomokuGame(BaseGame):
    """五子棋游戏 - 修复版本"""
    
    def __init__(self, board_size: int = 15, win_length: int = 5, **kwargs):
        self.board_size = board_size
        self.win_length = win_length
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self._last_move = None  # 记录最后一步，用于优化胜负判断
        super().__init__({'board_size': board_size, 'win_length': win_length})
    
    def reset(self) -> Dict[str, Any]:
        """重置游戏状态"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.game_state = config.GameState.ONGOING  # 修复：使用正确的枚举值
        self.move_count = 0
        self.history = []
        self._last_move = None
        
        return self.get_state()
    
    def step(self, action: Tuple[int, int]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行一步动作 - 修复版本
        
        Args:
            action: (row, col) 坐标
            
        Returns:
            observation: 观察状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        row, col = action
        
        # 检查动作有效性
        if not self._is_valid_action(action):
            return self.get_state(), -1, True, {'error': 'Invalid move', 'action': action}
        
        # 执行落子
        self.board[row, col] = self.current_player
        self._last_move = (row, col)
        self.history.append((self.current_player, action))
        self.move_count += 1
        
        # 检查游戏是否结束
        winner = None
        if self._check_win_at_position(row, col, self.current_player):
            winner = self.current_player
            self.game_state = config.GameState.FINISHED  # 修复：使用正确的枚举值
            
        elif self._is_board_full():
            self.game_state = config.GameState.FINISHED  # 修复：使用正确的枚举值
            
        # 计算奖励
        done = self.game_state == config.GameState.FINISHED  # 修复：使用正确的枚举值
        if done:
            if winner == self.current_player:
                reward = 1.0  # 获胜
            elif winner is not None:
                reward = -1.0  # 失败
            else:
                reward = 0.0  # 平局
        else:
            reward = 0.0  # 游戏继续
        
        info = {
            'winner': winner,
            'last_move': self._last_move,
            'move_count': self.move_count
        }
        
        # 切换玩家
        if not done:
            self.switch_player()
        
        return self.get_state(), reward, done, info
    
    def _check_win_at_position(self, row: int, col: int, player: int) -> bool:
        """
        检查指定位置是否形成获胜条件 - 优化版本
        只检查最后落子位置的四个方向
        """
        directions = [
            (0, 1),   # 水平
            (1, 0),   # 垂直  
            (1, 1),   # 主对角线
            (1, -1)   # 副对角线
        ]
        
        for dr, dc in directions:
            count = 1  # 当前位置算一个
            
            # 正方向计数
            r, c = row + dr, col + dc
            while (self._is_valid_position(r, c) and self.board[r, c] == player):
                count += 1
                r += dr
                c += dc
            
            # 负方向计数
            r, c = row - dr, col - dc
            while (self._is_valid_position(r, c) and self.board[r, c] == player):
                count += 1
                r -= dr
                c -= dc
            
            if count >= self.win_length:
                return True
        
        return False
    
    def _is_valid_position(self, row: int, col: int) -> bool:
        """检查位置是否在棋盘范围内"""
        return 0 <= row < self.board_size and 0 <= col < self.board_size
    
    def get_valid_actions(self, player: int = None) -> List[Tuple[int, int]]:
        """获取有效动作列表"""
        return [(i, j) for i in range(self.board_size) 
                for j in range(self.board_size) if self.board[i, j] == 0]
    
    def is_terminal(self) -> bool:
        """检查游戏是否结束 - 修复版本"""
        return self.game_state == config.GameState.FINISHED  # 修复：使用正确的枚举值
    
    def get_winner(self) -> Optional[int]:
        """获取获胜者 - 优化版本"""
        if not self.is_terminal():
            return None
            
        # 如果有最后一步记录，只检查该位置
        if self._last_move:
            row, col = self._last_move
            player = self.board[row, col]
            if player != 0 and self._check_win_at_position(row, col, player):
                return player
        
        # 如果没有最后一步记录，进行全盘检查（兼容性）
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] != 0:
                    player = self.board[i, j]
                    if self._check_win_at_position(i, j, player):
                        return player
        
        return None  # 平局
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前游戏状态"""
        return {
            'board': self.board.copy(),
            'current_player': self.current_player,
            'game_state': self.game_state,
            'move_count': self.move_count,
            'last_move': self._last_move
        }
    
    def render(self) -> np.ndarray:
        """渲染游戏画面"""
        return self.board.copy()
    
    def clone(self) -> 'GomokuGame':
        """克隆游戏状态"""
        import copy
        new_game = GomokuGame(self.board_size, self.win_length)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.game_state = self.game_state
        new_game.move_count = self.move_count
        new_game.history = copy.deepcopy(self.history)
        new_game._last_move = self._last_move
        return new_game
    
    def get_action_space(self):
        """获取动作空间"""
        return [(i, j) for i in range(self.board_size) for j in range(self.board_size)]
    
    def get_observation_space(self):
        """获取观察空间"""
        return {
            'board': (self.board_size, self.board_size),
            'current_player': 1,
            'valid_actions': []
        }
    
    def _is_valid_action(self, action: Tuple[int, int]) -> bool:
        """检查动作是否有效"""
        row, col = action
        return (self._is_valid_position(row, col) and self.board[row, col] == 0)
    
    def _is_board_full(self) -> bool:
        """检查棋盘是否已满"""
        return np.all(self.board != 0)
    
    def get_board_string(self) -> str:
        """获取棋盘字符串表示"""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        board_str = ""
        
        # 添加列标号
        board_str += "   " + " ".join([f"{i:2d}" for i in range(self.board_size)]) + "\n"
        
        for i in range(self.board_size):
            board_str += f"{i:2d} "
            for j in range(self.board_size):
                board_str += f" {symbols[self.board[i, j]]}"
            board_str += "\n"
        
        return board_str
    
    def print_board(self):
        """打印棋盘"""
        print(self.get_board_string())
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """获取合法移动（别名）"""
        return self.get_valid_actions()
    
    def update_game_state(self):
        """
        更新游戏状态 - 兼容性方法
        环境包装器可能需要这个方法
        """
        # 调用基类的更新方法
        super().update_game_state()
        
        # 检查游戏是否应该结束
        if self.move_count >= self.board_size * self.board_size:
            if self.game_state == config.GameState.ONGOING:
                self.game_state = config.GameState.FINISHED
    
    def get_game_info(self) -> Dict[str, Any]:
        """获取游戏信息"""
        info = super().get_game_info()
        info.update({
            'board_size': self.board_size,
            'win_length': self.win_length,
            'last_move': self._last_move,
            'board_full': self._is_board_full(),
            'winner': self.get_winner()
        })
        return info
