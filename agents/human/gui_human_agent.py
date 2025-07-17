"""
GUI人类智能体
处理图形界面中的人类玩家交互
"""

import time
from typing import Dict, List, Tuple, Any, Optional
from ..base_agent import BaseAgent


class GUIHumanAgent(BaseAgent):
    """GUI人类智能体 - 用于图形界面的人类玩家"""
    
    def __init__(self, name: str = "GUI Human", player_id: int = 1):
        super().__init__(name, player_id)
        self.pending_action = None
        self.waiting_for_input = False
    
    def get_action(self, observation: Any, env: Any) -> Any:
        """
        获取GUI人类玩家的动作
        这个方法在GUI中通常不会被直接调用，
        而是通过set_action方法设置动作
        """
        start_time = time.time()
        
        # 在GUI模式下，动作通常是通过GUI事件设置的
        if self.pending_action is not None:
            action = self.pending_action
            self.pending_action = None
            self.waiting_for_input = False
            
            # 更新统计
            move_time = time.time() - start_time
            self.total_moves += 1
            self.total_time += move_time
            
            return action
        
        # 如果没有pending action，标记为等待输入
        self.waiting_for_input = True
        return None
    
    def set_action(self, action: Any):
        """
        设置动作（由GUI事件调用）
        
        Args:
            action: GUI获取的动作
        """
        self.pending_action = action
        self.waiting_for_input = False
    
    def is_waiting_for_input(self) -> bool:
        """检查是否正在等待输入"""
        return self.waiting_for_input
    
    def clear_pending_action(self):
        """清除待处理的动作"""
        self.pending_action = None
        self.waiting_for_input = False
    
    def reset(self):
        """重置GUI人类智能体"""
        super().reset()
        self.pending_action = None
        self.waiting_for_input = False
    
    def get_info(self) -> Dict[str, Any]:
        """获取GUI人类智能体信息"""
        info = super().get_info()
        info.update({
            'type': 'GUI Human',
            'description': '图形界面人类玩家',
            'waiting_for_input': self.waiting_for_input,
            'has_pending_action': self.pending_action is not None
        })
        return info 
