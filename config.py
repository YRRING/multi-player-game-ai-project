# -*- coding:utf-8 -*-
###
# Created Date: Sunday, June 22nd 2025, 4:15:02 pm
# Author: Ying Wen
# -----
# Last Modified: 
# Modified By: 
# -----
# Copyright (c) 2025 MARL @ SJTU
###
"""
配置文件 - 修复版本
"""

from enum import Enum
from typing import Dict, Any


class GameState(Enum):
    """游戏状态枚举 - 修复版本"""
    ONGOING = "ongoing"
    FINISHED = "finished"  # 添加缺失的FINISHED状态
    PAUSED = "paused"
    TERMINATED = "terminated"


class PlayerType(Enum):
    """玩家类型枚举"""
    HUMAN = "human"
    AI = "ai"
    RANDOM = "random"


# 游戏配置
GAME_CONFIGS = {
    'gomoku': {
        'board_size': 15,
        'win_length': 5,
        'timeout': 300,  # 5分钟超时
        'max_moves': 225,  # 15*15棋盘最大步数
        'enable_undo': True,
        'enable_hint': True
    },
    'snake': {
        'board_size': 20,
        'initial_length': 3,
        'food_count': 5,
        'timeout': 300,
        'max_moves': 1000,
        'speed': 10  # 移动速度
    },
    'sokoban': {
        'timeout': 600,  # 10分钟超时
        'max_moves': 500,
        'enable_undo': True,
        'auto_save': True
    }
}

# AI配置
AI_CONFIGS = {
    'minimax': {
        'max_depth': 4,
        'use_alpha_beta': True,
        'evaluation_timeout': 5.0,
        'use_iterative_deepening': True,
        'use_transposition_table': True
    },
    'mcts': {
        'simulation_count': 1000,
        'exploration_constant': 1.414,
        'timeout': 10.0,
        'use_progressive_widening': True,
        'use_early_termination': True
    },
    'random': {
        'seed': None  # 随机种子
    },
    'rl': {
        'learning_rate': 0.001,
        'discount_factor': 0.99,
        'exploration_rate': 0.1,
        'batch_size': 32,
        'memory_size': 10000
    }
}

# 界面配置
UI_CONFIGS = {
    'window': {
        'width': 800,
        'height': 600,
        'title': '多人游戏AI框架',
        'resizable': True
    },
    'colors': {
        'background': '#FFFFFF',
        'foreground': '#000000',
        'highlight': '#0078D4',
        'error': '#FF0000',
        'success': '#008000'
    },
    'fonts': {
        'default_size': 12,
        'title_size': 16,
        'header_size': 14
    }
}

# 日志配置
LOG_CONFIGS = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'game_ai.log',
    'max_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# 性能配置
PERFORMANCE_CONFIGS = {
    'max_threads': 4,
    'enable_profiling': False,
    'cache_size': 1000,
    'memory_limit': 512 * 1024 * 1024,  # 512MB
    'timeout_warning': 30.0  # 30秒警告
}

# 调试配置
DEBUG_CONFIGS = {
    'enable_debug': False,
    'verbose_logging': False,
    'show_ai_thinking': False,
    'save_game_states': False,
    'enable_profiler': False
}

# 导出的常量
DEFAULT_BOARD_SIZE = 15
DEFAULT_WIN_LENGTH = 5
DEFAULT_TIMEOUT = 300
MAX_PLAYERS = 2
MIN_PLAYERS = 2

# 文件路径配置
PATHS = {
    'data': './data',
    'logs': './logs',
    'models': './models',
    'saves': './saves',
    'temp': './temp'
}

# 网络配置（如果需要在线功能）
NETWORK_CONFIGS = {
    'host': 'localhost',
    'port': 8080,
    'timeout': 30.0,
    'max_connections': 100,
    'enable_ssl': False
}

def get_game_config(game_type: str) -> Dict[str, Any]:
    """获取游戏配置"""
    return GAME_CONFIGS.get(game_type, {})

def get_ai_config(ai_type: str) -> Dict[str, Any]:
    """获取AI配置"""
    return AI_CONFIGS.get(ai_type, {})

def validate_config():
    """验证配置的有效性"""
    errors = []
    
    # 验证游戏配置
    for game_type, config in GAME_CONFIGS.items():
        if 'timeout' not in config:
            errors.append(f"游戏 {game_type} 缺少 timeout 配置")
        if 'max_moves' not in config:
            errors.append(f"游戏 {game_type} 缺少 max_moves 配置")
    
    # 验证AI配置
    for ai_type, config in AI_CONFIGS.items():
        if ai_type == 'minimax' and 'max_depth' not in config:
            errors.append(f"AI {ai_type} 缺少 max_depth 配置")
        if ai_type == 'mcts' and 'simulation_count' not in config:
            errors.append(f"AI {ai_type} 缺少 simulation_count 配置")
    
    if errors:
        raise ValueError("配置验证失败:\n" + "\n".join(errors))
    
    return True

# 初始化时验证配置
try:
    validate_config()
except ValueError as e:
    print(f"配置错误: {e}")
