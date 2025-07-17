"""
推箱子环境包装
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from games.base_env import BaseEnv
from .sokoban_game import SokobanGame

class SokobanEnv(BaseEnv):
    """推箱子环境"""
    
    def __init__(self, level_name: str = 'easy_1', mode: str = 'race', **kwargs):
        """
        初始化推箱子环境
        
        Args:
            level_name: 关卡名称
            mode: 游戏模式
        """
        self.level_name = level_name
        self.mode = mode
        
        # 创建游戏实例
        game = SokobanGame(level_name=level_name, mode=mode, **kwargs)
        super().__init__(game)
    
    def _setup_spaces(self) -> None:
        """设置观察空间和动作空间"""
        # 动作空间：4个方向
        self.action_space = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # 观察空间：两个网格
        obs_shape = self.game.get_observation_space()
        self.observation_space = obs_shape
    
    def _get_observation(self) -> Dict[str, Any]:
        """获取观察"""
        state = self.game.get_state()
        
        # 构建观察字典
        observation = {
            'grid1': state['grid1'],
            'grid2': state['grid2'],
            'player1_pos': state['player1_pos'],
            'player2_pos': state['player2_pos'],
            'boxes_in_place1': state['boxes_in_place1'],
            'boxes_in_place2': state['boxes_in_place2'],
            'moves1': state['moves1'],
            'moves2': state['moves2'],
            'current_player': state['current_player'],
            'target_boxes': state['target_boxes']
        }
        
        return observation
    
    def _get_action_mask(self) -> np.ndarray:
        """获取动作掩码"""
        valid_actions = self.game.get_valid_actions()
        mask = np.zeros(len(self.action_space), dtype=bool)
        
        for i, action in enumerate(self.action_space):
            if action in valid_actions:
                mask[i] = True
        
        return mask
    
    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """获取有效动作"""
        return self.game.get_valid_actions()
    
    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        return self.game.is_terminal()
    
    def get_winner(self) -> Optional[int]:
        """获取获胜者"""
        return self.game.get_winner()
    
    def render(self, mode='human') -> Optional[str]:
        """渲染环境"""
        if mode == 'human':
            print(self.game.render())
            return None
        elif mode == 'ansi':
            return self.game.render()
        return None
    
    def clone(self) -> 'SokobanEnv':
        """克隆环境"""
        cloned_env = SokobanEnv(self.level_name, self.mode)
        cloned_env.game = self.game.clone()
        return cloned_env
    
    def get_level_info(self) -> Dict[str, Any]:
        """获取关卡信息"""
        return {
            'level_name': self.level_name,
            'mode': self.mode,
            'target_boxes': self.game.original_level['target_boxes'],
            'max_moves': self.game.original_level.get('max_moves', 500),
            'difficulty': self.game.original_level.get('difficulty', 'unknown')
        }