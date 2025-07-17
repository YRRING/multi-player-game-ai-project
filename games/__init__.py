"""
游戏模块
"""

from .gomoku import GomokuGame, GomokuEnv
from .snake import SnakeGame, SnakeEnv

try:
    from .sokoban import SokobanGame, SokobanEnv
    _sokoban_games = ['SokobanGame', 'SokobanEnv']
except ImportError:
    _sokoban_games = []

__all__ = [
    'GomokuGame', 'GomokuEnv',
    'SnakeGame', 'SnakeEnv'
] + _sokoban_games
