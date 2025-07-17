"""
基础环境包装器 - 修复版本
为所有游戏提供统一的环境接口
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
import config


class BaseEnv(ABC):
    """
    基础环境类 - 所有游戏环境的抽象基类
    
    提供gym-style接口：
    - reset()
    - step()
    - render()
    - close()
    """
    
    def __init__(self, game):
        """
        初始化环境
        
        Args:
            game: 游戏实例
        """
        self.game = game
        self.observation_space = None
        self.action_space = None
        self._setup_spaces()
    
    @abstractmethod
    def _setup_spaces(self):
        """设置观察空间和动作空间"""
        pass
    
    @abstractmethod
    def _get_observation(self):
        """获取观察"""
        pass
    
    @abstractmethod
    def _get_action_mask(self):
        """获取动作掩码"""
        pass
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            
        Returns:
            (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 重置游戏
        self.game.reset()
        
        # 获取初始观察
        observation = self._get_observation()
        
        # 获取初始信息
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作 - 修复版本
        
        Args:
            action: 动作
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # 执行动作
        observation, reward, done, info = self.game.step(action)
        
        # 修复：移除不存在的方法调用
        # 原来的代码：self.game.update_game_state()
        # 这个方法在游戏类中不存在，所以直接删除
        
        # 获取新的观察
        observation = self._get_observation()
        
        # 分离terminated和truncated
        terminated = done and self.game.is_terminal()
        truncated = done and not self.game.is_terminal()
        
        # 更新信息
        info.update(self._get_info())
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """
        渲染环境
        
        Args:
            mode: 渲染模式
            
        Returns:
            渲染结果
        """
        return self.game.render()
    
    def close(self):
        """关闭环境"""
        pass
    
    def seed(self, seed: Optional[int] = None):
        """
        设置随机种子
        
        Args:
            seed: 随机种子
        """
        if seed is not None:
            np.random.seed(seed)
    
    def get_valid_actions(self) -> List[Any]:
        """获取有效动作"""
        return self.game.get_valid_actions()
    
    def is_terminal(self) -> bool:
        """检查是否终止"""
        return self.game.is_terminal()
    
    def get_winner(self) -> Optional[int]:
        """获取获胜者"""
        return self.game.get_winner()
    
    def clone(self) -> 'BaseEnv':
        """克隆环境"""
        cloned_game = self.game.clone()
        cloned_env = type(self)(cloned_game)
        return cloned_env
    
    def _get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        info = {
            'current_player': self.game.current_player,
            'move_count': self.game.move_count,
            'valid_actions': self.get_valid_actions(),
            'is_terminal': self.is_terminal()
        }
        
        # 添加游戏特定信息
        if hasattr(self.game, 'get_game_info'):
            info.update(self.game.get_game_info())
        
        return info
    
    def get_action_space(self) -> List[Any]:
        """获取动作空间"""
        if hasattr(self.game, 'get_action_space'):
            return self.game.get_action_space()
        return []
    
    def get_observation_space(self) -> Dict[str, Any]:
        """获取观察空间"""
        if hasattr(self.game, 'get_observation_space'):
            return self.game.get_observation_space()
        return {}
    
    def save_state(self) -> Dict[str, Any]:
        """保存环境状态"""
        return {
            'game_state': self.game.save_state() if hasattr(self.game, 'save_state') else self.game.get_state(),
            'env_info': self._get_info()
        }
    
    def load_state(self, state: Dict[str, Any]):
        """加载环境状态"""
        if 'game_state' in state:
            if hasattr(self.game, 'load_state'):
                self.game.load_state(state['game_state'])
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(game={self.game.__class__.__name__})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"{self.__class__.__name__}(game={self.game})"
