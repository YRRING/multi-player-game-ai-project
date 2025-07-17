"""
智能体模块
"""

from .base_agent import BaseAgent
from .human.human_agent import HumanAgent
from .human.gui_human_agent import GUIHumanAgent
from .ai_bots.random_bot import RandomBot
from .ai_bots.minimax_bot import MinimaxBot
from .ai_bots.mcts_bot import MCTSBot
from .ai_bots.rl_bot import RLBot
from .ai_bots.behavior_tree_bot import BehaviorTreeBot
from .ai_bots.snake_ai import SnakeAI, SmartSnakeAI

# 检查是否有推箱子AI（如果已经实现）
try:
    from .ai_bots.sokoban_ai import SokobanAI, SmartSokobanAI, ExpertSokobanAI
    _sokoban_agents = ['SokobanAI', 'SmartSokobanAI', 'ExpertSokobanAI']
except ImportError:
    _sokoban_agents = []

__all__ = [
    'BaseAgent',
    'HumanAgent', 
    'GUIHumanAgent',
    'RandomBot',
    'MinimaxBot',
    'MCTSBot',
    'RLBot',
    'BehaviorTreeBot',
    'SnakeAI',
    'SmartSnakeAI'
] + _sokoban_agents
