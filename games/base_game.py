"""
基础游戏类 - 完整修复版本
提供所有游戏模块的通用基础功能
"""

import time
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
import config


class BaseGame(ABC):
    """
    基础游戏类 - 所有游戏的抽象基类
    
    提供游戏的通用功能：
    - 状态管理
    - 玩家切换
    - 历史记录
    - 时间管理
    - 基础验证
    """
    
    def __init__(self, game_config: Dict[str, Any] = None):
        """
        初始化基础游戏
        
        Args:
            game_config: 游戏配置字典
        """
        self.game_config = game_config or {}
        
        # 核心状态
        self.current_player = 1
        self.game_state = config.GameState.ONGOING
        self.move_count = 0
        self.history = []
        
        # 时间管理
        self.start_time = time.time()
        self.last_move_time = time.time()
        self.timeout = self.game_config.get('timeout', config.DEFAULT_TIMEOUT)
        self.max_moves = self.game_config.get('max_moves', 1000)
        
        # 游戏统计
        self.stats = {
            'player1_moves': 0,
            'player2_moves': 0,
            'total_time': 0,
            'average_move_time': 0
        }
        
        # 调试和日志
        self.debug_mode = self.game_config.get('debug', False)
        self.verbose_logging = self.game_config.get('verbose', False)
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        重置游戏状态
        
        Returns:
            初始游戏状态
        """
        self.current_player = 1
        self.game_state = config.GameState.ONGOING
        self.move_count = 0
        self.history = []
        self.start_time = time.time()
        self.last_move_time = time.time()
        
        # 重置统计
        self.stats = {
            'player1_moves': 0,
            'player2_moves': 0,
            'total_time': 0,
            'average_move_time': 0
        }
        
        return self.get_state()
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 动作
            
        Returns:
            (observation, reward, done, info)
        """
        pass
    
    @abstractmethod
    def get_valid_actions(self, player: int = None) -> List[Any]:
        """
        获取有效动作列表
        
        Args:
            player: 玩家ID，None表示当前玩家
            
        Returns:
            有效动作列表
        """
        pass
    
    def is_terminal(self) -> bool:
        """
        检查游戏是否结束
        
        Returns:
            是否终止
        """
        # 检查游戏状态
        if self.game_state == config.GameState.FINISHED:
            return True
        
        # 检查超时
        if self.timeout and (time.time() - self.start_time) > self.timeout:
            self.game_state = config.GameState.TERMINATED
            return True
        
        # 检查最大移动次数
        if self.move_count >= self.max_moves:
            self.game_state = config.GameState.TERMINATED
            return True
        
        return False
    
    @abstractmethod
    def get_winner(self) -> Optional[int]:
        """
        获取获胜者
        
        Returns:
            获胜者ID，None表示平局或游戏未结束
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        获取当前游戏状态
        
        Returns:
            游戏状态字典
        """
        pass
    
    @abstractmethod
    def render(self) -> Any:
        """
        渲染游戏画面
        
        Returns:
            渲染结果
        """
        pass
    
    @abstractmethod
    def clone(self) -> 'BaseGame':
        """
        克隆游戏状态
        
        Returns:
            克隆的游戏实例
        """
        pass
    
    def switch_player(self):
        """切换当前玩家"""
        old_player = self.current_player
        self.current_player = 3 - self.current_player
        
        if self.debug_mode:
            print(f"Player switched from {old_player} to {self.current_player}")
    
    def record_move(self, player: int, action: Any, info: Dict[str, Any] = None):
        """
        记录移动
        
        Args:
            player: 玩家ID
            action: 动作
            info: 额外信息
        """
        current_time = time.time()
        move_time = current_time - self.last_move_time
        
        move_record = {
            'player': player,
            'action': action,
            'move_count': self.move_count,
            'timestamp': current_time - self.start_time,
            'move_time': move_time,
            'info': info or {}
        }
        
        self.history.append(move_record)
        self.last_move_time = current_time
        
        # 更新统计
        if player == 1:
            self.stats['player1_moves'] += 1
        else:
            self.stats['player2_moves'] += 1
        
        self.stats['total_time'] = current_time - self.start_time
        self.stats['average_move_time'] = self.stats['total_time'] / max(1, self.move_count)
        
        if self.verbose_logging:
            print(f"Move recorded: Player {player}, Action {action}, Time {move_time:.2f}s")
    
    def get_game_info(self) -> Dict[str, Any]:
        """
        获取游戏信息
        
        Returns:
            游戏信息字典
        """
        current_time = time.time()
        
        return {
            'current_player': self.current_player,
            'game_state': self.game_state.value,
            'move_count': self.move_count,
            'game_time': current_time - self.start_time,
            'history_length': len(self.history),
            'stats': self.stats.copy(),
            'timeout': self.timeout,
            'max_moves': self.max_moves,
            'time_remaining': max(0, self.timeout - (current_time - self.start_time)) if self.timeout else None
        }
    
    def get_move_history(self, player: int = None) -> List[Dict[str, Any]]:
        """
        获取移动历史
        
        Args:
            player: 玩家ID，None表示所有玩家
            
        Returns:
            移动历史列表
        """
        if player is None:
            return self.history.copy()
        else:
            return [move for move in self.history if move['player'] == player]
    
    def undo_last_move(self) -> bool:
        """
        撤销最后一步移动
        
        Returns:
            是否成功撤销
        """
        if not self.history:
            return False
        
        # 简单实现：只移除历史记录，具体游戏需要重写
        last_move = self.history.pop()
        self.move_count -= 1
        
        # 更新统计
        if last_move['player'] == 1:
            self.stats['player1_moves'] -= 1
        else:
            self.stats['player2_moves'] -= 1
        
        if self.debug_mode:
            print(f"Undone move: {last_move}")
        
        return True
    
    def pause_game(self):
        """暂停游戏"""
        if self.game_state == config.GameState.ONGOING:
            self.game_state = config.GameState.PAUSED
            if self.debug_mode:
                print("Game paused")
    
    def resume_game(self):
        """恢复游戏"""
        if self.game_state == config.GameState.PAUSED:
            self.game_state = config.GameState.ONGOING
            if self.debug_mode:
                print("Game resumed")
    
    def terminate_game(self, reason: str = "Manual termination"):
        """
        终止游戏
        
        Args:
            reason: 终止原因
        """
        self.game_state = config.GameState.TERMINATED
        self.record_move(0, "TERMINATE", {'reason': reason})
        
        if self.debug_mode:
            print(f"Game terminated: {reason}")
    
    def validate_action(self, action: Any, player: int = None) -> bool:
        """
        验证动作是否有效
        
        Args:
            action: 动作
            player: 玩家ID
            
        Returns:
            是否有效
        """
        if player is None:
            player = self.current_player
        
        valid_actions = self.get_valid_actions(player)
        return action in valid_actions
    
    def get_action_space(self) -> List[Any]:
        """
        获取动作空间
        
        Returns:
            所有可能的动作列表
        """
        # 默认实现，子类应该重写
        return []
    
    def get_observation_space(self) -> Dict[str, Any]:
        """
        获取观察空间
        
        Returns:
            观察空间描述
        """
        # 默认实现，子类应该重写
        return {}
    
    def get_reward_info(self) -> Dict[str, Any]:
        """
        获取奖励信息
        
        Returns:
            奖励计算相关信息
        """
        return {
            'reward_type': 'terminal',  # 默认只在终止时给奖励
            'max_reward': 1.0,
            'min_reward': -1.0,
            'step_reward': 0.0
        }
    
    def save_state(self) -> Dict[str, Any]:
        """
        保存游戏状态
        
        Returns:
            可序列化的状态字典
        """
        return {
            'current_player': self.current_player,
            'game_state': self.game_state.value,
            'move_count': self.move_count,
            'history': self.history.copy(),
            'stats': self.stats.copy(),
            'start_time': self.start_time,
            'last_move_time': self.last_move_time
        }
    
    def update_game_state(self):
        """
        更新游戏状态 - 兼容性方法
        某些环境包装器可能需要这个方法
        """
        # 检查是否应该终止游戏
        if self.is_terminal():
            if self.game_state == config.GameState.ONGOING:
                self.game_state = config.GameState.FINISHED
        
        # 更新统计信息
        current_time = time.time()
        self.stats['total_time'] = current_time - self.start_time
        if self.move_count > 0:
            self.stats['average_move_time'] = self.stats['total_time'] / self.move_count
    
    def load_state(self, state: Dict[str, Any]):
        """
        加载游戏状态
        
        Args:
            state: 状态字典
        """
        self.current_player = state.get('current_player', 1)
        self.game_state = config.GameState(state.get('game_state', 'ongoing'))
        self.move_count = state.get('move_count', 0)
        self.history = state.get('history', [])
        self.stats = state.get('stats', {})
        self.start_time = state.get('start_time', time.time())
        self.last_move_time = state.get('last_move_time', time.time())
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(player={self.current_player}, moves={self.move_count}, state={self.game_state.value})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"{self.__class__.__name__}(current_player={self.current_player}, game_state={self.game_state}, move_count={self.move_count})"
