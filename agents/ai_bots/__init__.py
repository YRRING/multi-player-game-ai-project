"""
AI Bot模块
"""

from .random_bot import RandomBot
from .minimax_bot import MinimaxBot
from .mcts_bot import MCTSBot
from .rl_bot import RLBot
from .behavior_tree_bot import BehaviorTreeBot
from .snake_ai import SnakeAI, SmartSnakeAI

# 检查是否有推箱子AI
try:
    from .sokoban_ai import SokobanAI, SmartSokobanAI, ExpertSokobanAI
    _sokoban_bots = ['SokobanAI', 'SmartSokobanAI', 'ExpertSokobanAI']
except ImportError:
    _sokoban_bots = []

__all__ = [
    'RandomBot',
    'MinimaxBot',
    'MCTSBot',
    'RLBot',
    'BehaviorTreeBot',
    'SnakeAI',
    'SmartSnakeAI'
] + _sokoban_bots
