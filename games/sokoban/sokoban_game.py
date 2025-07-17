"""
推箱子游戏逻辑
"""

import json
import os
import numpy as np
import copy
import time
from typing import Dict, List, Tuple, Any, Optional
from games.base_game import BaseGame

# 游戏元素定义
EMPTY = 0      # 空地
WALL = 1       # 墙
PLAYER = 2     # 玩家
BOX = 3        # 箱子
TARGET = 4     # 目标位置
BOX_ON_TARGET = 5  # 箱子在目标位置上
PLAYER_ON_TARGET = 6  # 玩家在目标位置上

# 字符到数字的映射
CHAR_TO_NUM = {
    ' ': EMPTY,
    '#': WALL,
    '@': PLAYER,
    '$': BOX,
    '.': TARGET,
    '*': BOX_ON_TARGET,
    '+': PLAYER_ON_TARGET
}

# 数字到字符的映射
NUM_TO_CHAR = {v: k for k, v in CHAR_TO_NUM.items()}

def _load_levels_from_json() -> Dict[str, Any]:
    """从JSON文件加载关卡数据"""
    try:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, 'levels.json')
        
        # 如果JSON文件不存在，返回默认关卡
        if not os.path.exists(json_path):
            return _get_default_levels()
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('levels', {})
    except Exception as e:
        print(f"加载关卡文件失败: {e}")
        return _get_default_levels()

def _get_default_levels() -> Dict[str, Any]:
    """获取默认关卡数据（作为fallback）"""
    return {
        'easy_1': {
            'name': 'Easy Level 1',
            'difficulty': 'easy',
            'layout': [
                "    #####",
                "    #   #",
                "    #$  #",
                "  ###  $##",
                "  #  $ $ #",
                "### # ## #   ######",
                "#   # ## #####  ..#",
                "# $  $          ..#",
                "##### ### #@##  ..#",
                "    #     #########",
                "    #######"
            ],
            'target_boxes': 3,
            'max_moves': 100
        },
        'easy_2': {
            'name': 'Easy Level 2',
            'difficulty': 'easy',
            'layout': [
                "############",
                "#..  #     ###",
                "#..  # $  $  #",
                "#..  #$####  #",
                "#..    @ ##  #",
                "#..  # #  $ ##",
                "###### ##$ $ #",
                "  # $  $ $ $ #",
                "  #    #     #",
                "  ############"
            ],
            'target_boxes': 4,
            'max_moves': 150
        },
        'medium_1': {
            'name': 'Medium Level 1',
            'difficulty': 'medium',
            'layout': [
                "        ########",
                "        #     @#",
                "        # $#$ ##",
                "        # $  $#",
                "        ##$ $ #",
                "######### $ # ###",
                "#....  ## $  $  #",
                "##...    $  $   #",
                "#....  ##########",
                "########"
            ],
            'target_boxes': 5,
            'max_moves': 200
        },
        'hard_1': {
            'name': 'Hard Level 1',
            'difficulty': 'hard',
            'layout': [
                "          ######",
                "          #..  #",
                "          #..  #",
                "      #####..  #",
                "      #       #",
                "      #  $$ ##",
                "      # $ $ $ #",
                "#### ##$ $ $  #",
                "#. @  $ $ $ ###",
                "#.    $ $ $  #",
                "#.    ##     #",
                "######### ###"
            ],
            'target_boxes': 6,
            'max_moves': 300
        }
    }

# 全局关卡数据
LEVELS = _load_levels_from_json()

def create_level(layout: List[str], target_boxes: int = None, max_moves: int = 200) -> Dict[str, Any]:
    """
    创建关卡数据
    
    Args:
        layout: 关卡布局字符串列表
        target_boxes: 目标箱子数量
        max_moves: 最大移动次数
        
    Returns:
        关卡数据字典
    """
    # 转换为数字数组
    max_width = max(len(row) for row in layout)
    grid = np.zeros((len(layout), max_width), dtype=int)
    
    box_count = 0
    target_count = 0
    player_count = 0
    
    for i, row in enumerate(layout):
        for j, char in enumerate(row):
            if char in CHAR_TO_NUM:
                grid[i, j] = CHAR_TO_NUM[char]
                
                if char == '$':
                    box_count += 1
                elif char == '*':
                    box_count += 1
                elif char == '.':
                    target_count += 1
                elif char == '+':
                    target_count += 1
                elif char == '@':
                    player_count += 1
    
    # 如果没有指定目标箱子数，使用检测到的数量
    if target_boxes is None:
        target_boxes = box_count
    
    level_data = {
        'name': f'Custom Level',
        'difficulty': 'custom',
        'layout': layout,
        'grid': grid,
        'target_boxes': target_boxes,
        'max_moves': max_moves,
        'box_count': box_count,
        'target_count': target_count,
        'player_count': player_count
    }
    
    return level_data

def validate_level(level_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    验证关卡数据是否有效
    
    Args:
        level_data: 关卡数据
        
    Returns:
        (是否有效, 错误信息)
    """
    try:
        layout = level_data['layout']
        grid = level_data.get('grid')
        
        if grid is None:
            # 重新生成grid
            level_data = create_level(layout, level_data.get('target_boxes'))
            grid = level_data['grid']
        
        # 检查基本要素
        if level_data['player_count'] != 1:
            return False, f"关卡必须包含且仅包含一个玩家，当前有{level_data['player_count']}个"
        
        if level_data['box_count'] == 0:
            return False, "关卡必须包含至少一个箱子"
        
        if level_data['target_count'] == 0:
            return False, "关卡必须包含至少一个目标位置"
        
        if level_data['box_count'] > level_data['target_count']:
            return False, f"箱子数量({level_data['box_count']})不能超过目标位置数量({level_data['target_count']})"
        
        # 检查是否有被墙包围的区域
        if not _check_reachability(grid):
            return False, "关卡存在无法到达的区域"
        
        return True, "关卡验证通过"
        
    except Exception as e:
        return False, f"关卡验证失败: {str(e)}"

def _check_reachability(grid: np.ndarray) -> bool:
    """检查关卡的可达性"""
    # 找到玩家位置
    player_pos = None
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] in [PLAYER, PLAYER_ON_TARGET]:
                player_pos = (i, j)
                break
        if player_pos:
            break
    
    if not player_pos:
        return False
    
    # 使用BFS检查可达性
    visited = set()
    queue = [player_pos]
    visited.add(player_pos)
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        row, col = queue.pop(0)
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < grid.shape[0] and 
                0 <= new_col < grid.shape[1] and
                (new_row, new_col) not in visited):
                
                cell = grid[new_row, new_col]
                
                # 如果不是墙，可以到达
                if cell != WALL:
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col))
    
    # 检查所有箱子和目标位置是否可达
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] in [BOX, TARGET, BOX_ON_TARGET]:
                if (i, j) not in visited:
                    return False
    
    return True

def get_level_by_difficulty(difficulty: str) -> List[str]:
    """根据难度获取关卡列表"""
    return [key for key, level in LEVELS.items() if level['difficulty'] == difficulty]

def get_random_level() -> str:
    """获取随机关卡"""
    import random
    return random.choice(list(LEVELS.keys()))

class SokobanGame(BaseGame):
    """
    推箱子游戏类 - 双人对战版本
    两个玩家同时在不同区域推箱子，比较完成速度和效率
    """
    
    def __init__(self, level_name: str = 'easy_1', mode: str = 'race', **kwargs):
        """
        初始化推箱子游戏
        
        Args:
            level_name: 关卡名称
            mode: 游戏模式 ('race': 竞速模式, 'score': 计分模式)
        """
        self.level_name = level_name
        self.mode = mode
        self.original_level = LEVELS[level_name]
        
        # 游戏状态
        self.grid1 = None  # 玩家1的游戏区域
        self.grid2 = None  # 玩家2的游戏区域
        self.player1_pos = None
        self.player2_pos = None
        self.boxes_in_place1 = 0  # 玩家1放置好的箱子数
        self.boxes_in_place2 = 0  # 玩家2放置好的箱子数
        self.moves1 = 0  # 玩家1移动次数
        self.moves2 = 0  # 玩家2移动次数
        
        # 游戏结果
        self.winner = None
        self.game_over = False
        
        # 继承基类初始化
        super().__init__(kwargs)
    
    def reset(self) -> Dict[str, Any]:
        """重置游戏状态"""
        self.current_player = 1
        self.move_count = 0
        self.start_time = time.time()
        self.history = []
        self.game_over = False
        self.winner = None
        
        # 创建两个相同的游戏区域
        level_data = create_level(self.original_level['layout'], 
                                self.original_level['target_boxes'])
        
        self.grid1 = level_data['grid'].copy()
        self.grid2 = level_data['grid'].copy()
        
        # 找到玩家初始位置
        self.player1_pos = self._find_player_position(self.grid1)
        self.player2_pos = self._find_player_position(self.grid2)
        
        # 初始化统计
        self.boxes_in_place1 = self._count_boxes_in_place(self.grid1)
        self.boxes_in_place2 = self._count_boxes_in_place(self.grid2)
        self.moves1 = 0
        self.moves2 = 0
        
        return self.get_state()
    
    def step(self, action: Tuple[int, int]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行一步移动
        
        Args:
            action: 移动方向 (dr, dc)
            
        Returns:
            observation: 游戏状态
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        if self.game_over:
            return self.get_state(), 0, True, {'error': 'Game already over'}
        
        # 执行移动
        success, info = self._move_player(self.current_player, action)
        
        # 计算奖励
        reward = self._calculate_reward(self.current_player, success, info)
        
        # 检查游戏是否结束
        self._check_game_over()
        
        # 记录移动
        self.record_move(self.current_player, action, {
            'success': success,
            'info': info,
            'reward': reward
        })
        
        # 切换玩家
        self.switch_player()
        
        return self.get_state(), reward, self.game_over, info
    
    def _move_player(self, player: int, action: Tuple[int, int]) -> Tuple[bool, Dict[str, Any]]:
        """
        移动玩家
        
        Args:
            player: 玩家编号
            action: 移动方向
            
        Returns:
            (是否成功, 信息)
        """
        dr, dc = action
        
        if player == 1:
            grid = self.grid1
            player_pos = self.player1_pos
        else:
            grid = self.grid2
            player_pos = self.player2_pos
        
        current_row, current_col = player_pos
        new_row, new_col = current_row + dr, current_col + dc
        
        # 检查边界
        if (new_row < 0 or new_row >= grid.shape[0] or 
            new_col < 0 or new_col >= grid.shape[1]):
            return False, {'error': 'Out of bounds'}
        
        # 检查目标位置
        target_cell = grid[new_row, new_col]
        
        if target_cell == WALL:
            return False, {'error': 'Cannot move into wall'}
        
        # 如果目标位置有箱子
        if target_cell in [BOX, BOX_ON_TARGET]:
            # 检查箱子是否可以推动
            box_new_row, box_new_col = new_row + dr, new_col + dc
            
            # 检查箱子推动后的位置
            if (box_new_row < 0 or box_new_row >= grid.shape[0] or 
                box_new_col < 0 or box_new_col >= grid.shape[1]):
                return False, {'error': 'Cannot push box out of bounds'}
            
            box_target_cell = grid[box_new_row, box_new_col]
            
            if box_target_cell in [WALL, BOX, BOX_ON_TARGET]:
                return False, {'error': 'Cannot push box into obstacle'}
            
            # 推动箱子
            self._push_box(player, (new_row, new_col), (box_new_row, box_new_col))
            box_pushed = True
        else:
            box_pushed = False
        
        # 移动玩家
        self._move_player_to(player, (new_row, new_col))
        
        # 更新移动计数
        if player == 1:
            self.moves1 += 1
        else:
            self.moves2 += 1
        
        return True, {
            'box_pushed': box_pushed,
            'new_position': (new_row, new_col)
        }
    
    def _push_box(self, player: int, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """推动箱子"""
        if player == 1:
            grid = self.grid1
        else:
            grid = self.grid2
        
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # 检查原位置是否在目标上
        was_on_target = grid[from_row, from_col] == BOX_ON_TARGET
        
        # 检查新位置是否是目标
        is_target = grid[to_row, to_col] == TARGET
        
        # 更新原位置
        if was_on_target:
            grid[from_row, from_col] = TARGET
        else:
            grid[from_row, from_col] = EMPTY
        
        # 更新新位置
        if is_target:
            grid[to_row, to_col] = BOX_ON_TARGET
        else:
            grid[to_row, to_col] = BOX
        
        # 更新统计
        if player == 1:
            self.boxes_in_place1 = self._count_boxes_in_place(self.grid1)
        else:
            self.boxes_in_place2 = self._count_boxes_in_place(self.grid2)
    
    def _move_player_to(self, player: int, new_pos: Tuple[int, int]):
        """移动玩家到新位置"""
        if player == 1:
            grid = self.grid1
            old_pos = self.player1_pos
        else:
            grid = self.grid2
            old_pos = self.player2_pos
        
        old_row, old_col = old_pos
        new_row, new_col = new_pos
        
        # 检查原位置是否在目标上
        was_on_target = grid[old_row, old_col] == PLAYER_ON_TARGET
        
        # 检查新位置是否是目标
        is_target = grid[new_row, new_col] == TARGET
        
        # 更新原位置
        if was_on_target:
            grid[old_row, old_col] = TARGET
        else:
            grid[old_row, old_col] = EMPTY
        
        # 更新新位置
        if is_target:
            grid[new_row, new_col] = PLAYER_ON_TARGET
        else:
            grid[new_row, new_col] = PLAYER
        
        # 更新玩家位置
        if player == 1:
            self.player1_pos = new_pos
        else:
            self.player2_pos = new_pos
    
    def _find_player_position(self, grid: np.ndarray) -> Tuple[int, int]:
        """找到玩家位置"""
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] in [PLAYER, PLAYER_ON_TARGET]:
                    return (i, j)
        return None
    
    def _count_boxes_in_place(self, grid: np.ndarray) -> int:
        """计算放置好的箱子数量"""
        count = 0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == BOX_ON_TARGET:
                    count += 1
        return count
    
    def _calculate_reward(self, player: int, success: bool, info: Dict[str, Any]) -> float:
        """计算奖励"""
        if not success:
            return -1  # 无效移动惩罚
        
        reward = 0
        
        # 基本移动奖励
        reward += 1
        
        # 推箱子奖励
        if info.get('box_pushed', False):
            reward += 10
        
        # 箱子放置奖励
        if player == 1:
            new_boxes = self.boxes_in_place1
        else:
            new_boxes = self.boxes_in_place2
        
        old_boxes = info.get('old_boxes_in_place', 0)
        if new_boxes > old_boxes:
            reward += 50 * (new_boxes - old_boxes)
        
        # 完成关卡奖励
        if self._is_level_complete(player):
            reward += 1000
        
        return reward
    
    def _is_level_complete(self, player: int) -> bool:
        """检查关卡是否完成"""
        if player == 1:
            return self.boxes_in_place1 >= self.original_level['target_boxes']
        else:
            return self.boxes_in_place2 >= self.original_level['target_boxes']
    
    def _check_game_over(self):
        """检查游戏是否结束"""
        player1_complete = self._is_level_complete(1)
        player2_complete = self._is_level_complete(2)
        
        if self.mode == 'race':
            # 竞速模式：第一个完成的玩家获胜
            if player1_complete and player2_complete:
                # 同时完成，比较移动次数
                if self.moves1 < self.moves2:
                    self.winner = 1
                elif self.moves2 < self.moves1:
                    self.winner = 2
                else:
                    self.winner = None  # 平局
                self.game_over = True
            elif player1_complete:
                self.winner = 1
                self.game_over = True
            elif player2_complete:
                self.winner = 2
                self.game_over = True
        
        # 检查最大移动次数
        max_moves = self.original_level.get('max_moves', 500)
        if self.moves1 >= max_moves or self.moves2 >= max_moves:
            # 比较已完成的箱子数
            if self.boxes_in_place1 > self.boxes_in_place2:
                self.winner = 1
            elif self.boxes_in_place2 > self.boxes_in_place1:
                self.winner = 2
            else:
                self.winner = None  # 平局
            self.game_over = True
    
    def get_valid_actions(self, player: int = None) -> List[Tuple[int, int]]:
        """获取有效动作"""
        if player is None:
            player = self.current_player
        
        actions = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
        
        if player == 1:
            grid = self.grid1
            pos = self.player1_pos
        else:
            grid = self.grid2
            pos = self.player2_pos
        
        if pos is None:
            return actions
        
        row, col = pos
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # 检查边界
            if (new_row < 0 or new_row >= grid.shape[0] or 
                new_col < 0 or new_col >= grid.shape[1]):
                continue
            
            target_cell = grid[new_row, new_col]
            
            if target_cell == WALL:
                continue
            
            # 如果是箱子，检查是否可以推动
            if target_cell in [BOX, BOX_ON_TARGET]:
                box_new_row, box_new_col = new_row + dr, new_col + dc
                
                if (box_new_row < 0 or box_new_row >= grid.shape[0] or 
                    box_new_col < 0 or box_new_col >= grid.shape[1]):
                    continue
                
                box_target_cell = grid[box_new_row, box_new_col]
                
                if box_target_cell in [WALL, BOX, BOX_ON_TARGET]:
                    continue
            
            actions.append((dr, dc))
        
        return actions
    
    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        return self.game_over
    
    def get_winner(self) -> Optional[int]:
        """获取获胜者"""
        return self.winner
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前游戏状态"""
        return {
            'grid1': self.grid1.copy() if self.grid1 is not None else None,
            'grid2': self.grid2.copy() if self.grid2 is not None else None,
            'player1_pos': self.player1_pos,
            'player2_pos': self.player2_pos,
            'boxes_in_place1': self.boxes_in_place1,
            'boxes_in_place2': self.boxes_in_place2,
            'moves1': self.moves1,
            'moves2': self.moves2,
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner,
            'level_name': self.level_name,
            'target_boxes': self.original_level['target_boxes']
        }
    
    def render(self) -> str:
        """渲染游戏状态"""
        if self.grid1 is None or self.grid2 is None:
            return "Game not initialized"
        
        # 字符映射
        char_map = {
            EMPTY: ' ',
            WALL: '#',
            PLAYER: '@',
            BOX: '$',
            TARGET: '.',
            BOX_ON_TARGET: '*',
            PLAYER_ON_TARGET: '+'
        }
        
        output = []
        output.append(f"推箱子游戏 - 关卡: {self.level_name}")
        output.append(f"目标箱子数: {self.original_level['target_boxes']}")
        output.append("")
        
        # 显示玩家1区域
        output.append("玩家1区域:")
        output.append(f"已放置箱子: {self.boxes_in_place1}/{self.original_level['target_boxes']}")
        output.append(f"移动次数: {self.moves1}")
        output.append("")
        
        for row in self.grid1:
            line = ""
            for cell in row:
                line += char_map.get(cell, '?')
            output.append(line)
        
        output.append("")
        output.append("=" * 20)
        output.append("")
        
        # 显示玩家2区域
        output.append("玩家2区域:")
        output.append(f"已放置箱子: {self.boxes_in_place2}/{self.original_level['target_boxes']}")
        output.append(f"移动次数: {self.moves2}")
        output.append("")
        
        for row in self.grid2:
            line = ""
            for cell in row:
                line += char_map.get(cell, '?')
            output.append(line)
        
        if self.game_over:
            output.append("")
            if self.winner:
                output.append(f"游戏结束！获胜者: 玩家{self.winner}")
            else:
                output.append("游戏结束！平局")
        
        return "\n".join(output)
    
    def clone(self) -> 'SokobanGame':
        """克隆游戏状态"""
        cloned = SokobanGame(self.level_name, self.mode)
        cloned.grid1 = self.grid1.copy() if self.grid1 is not None else None
        cloned.grid2 = self.grid2.copy() if self.grid2 is not None else None
        cloned.player1_pos = self.player1_pos
        cloned.player2_pos = self.player2_pos
        cloned.boxes_in_place1 = self.boxes_in_place1
        cloned.boxes_in_place2 = self.boxes_in_place2
        cloned.moves1 = self.moves1
        cloned.moves2 = self.moves2
        cloned.current_player = self.current_player
        cloned.game_over = self.game_over
        cloned.winner = self.winner
        cloned.move_count = self.move_count
        cloned.history = self.history.copy()
        return cloned
    
    def get_action_space(self) -> List[Tuple[int, int]]:
        """获取动作空间"""
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
    
    def get_observation_space(self) -> Tuple[int, int, int]:
        """获取观察空间"""
        if self.grid1 is not None:
            return (self.grid1.shape[0], self.grid1.shape[1], 2)  # 两个网格
        return (12, 12, 2)  # 默认大小