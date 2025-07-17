"""
Minimax Bot - 完善优化版本
使用alpha-beta剪枝的Minimax算法，增强评估函数和优化策略
"""

import time
import random
from typing import Dict, List, Tuple, Any, Optional
from agents.base_agent import BaseAgent
import config
import numpy as np


class MinimaxBot(BaseAgent):
    """
    Minimax Bot - 完善的alpha-beta剪枝实现
    增强的评估函数和搜索优化
    """
    
    def __init__(self, name: str = "MinimaxBot", player_id: int = 1, 
                 max_depth: int = 4, time_limit: float = 5.0):
        super().__init__(name, player_id)
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.nodes_evaluated = 0
        self.pruning_count = 0
        
        # 从配置获取参数
        ai_config = config.AI_CONFIGS.get('minimax', {})
        self.max_depth = ai_config.get('max_depth', max_depth)
        self.use_alpha_beta = ai_config.get('use_alpha_beta', True)
        self.evaluation_timeout = ai_config.get('evaluation_timeout', time_limit)
        
        # 缓存评估结果
        self.evaluation_cache = {}
        self.cache_hits = 0
        
        # 增强参数
        self.use_iterative_deepening = True
        self.use_transposition_table = True
        self.killer_moves = {}  # 杀手启发
        self.history_heuristic = {}  # 历史启发
        
    def get_action(self, observation: Any, env: Any) -> Any:
        """
        使用Minimax算法选择最佳动作
        """
        start_time = time.time()
        
        # 重置统计
        self.nodes_evaluated = 0
        self.pruning_count = 0
        self.cache_hits = 0
        
        # 获取有效动作
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            return None
        
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        # 预处理动作排序
        ordered_actions = self._order_actions(valid_actions, env)
        
        best_action = None
        best_value = float('-inf')
        
        # 使用迭代加深搜索
        if self.use_iterative_deepening:
            for depth in range(1, self.max_depth + 1):
                if time.time() - start_time > self.evaluation_timeout * 0.8:
                    break
                
                current_best = self._search_depth(env, ordered_actions, depth, start_time)
                if current_best[0] is not None:
                    best_action, best_value = current_best
        else:
            best_action, best_value = self._search_depth(env, ordered_actions, self.max_depth, start_time)
        
        # 更新统计
        move_time = time.time() - start_time
        self.total_moves += 1
        self.total_time += move_time
        
        return best_action if best_action else random.choice(valid_actions)
    
    def _search_depth(self, env: Any, actions: List[Any], depth: int, start_time: float) -> Tuple[Any, float]:
        """在指定深度搜索最佳动作"""
        best_action = None
        best_value = float('-inf')
        
        for action in actions:
            if time.time() - start_time > self.evaluation_timeout:
                break
            
            # 克隆环境并执行动作
            game_copy = self._safe_clone(env)
            if game_copy is None:
                continue
                
            game_copy.step(action)
            
            # 使用alpha-beta剪枝评估
            if self.use_alpha_beta:
                value = self._minimax_alpha_beta(
                    game_copy, depth - 1, float('-inf'), float('inf'), 
                    False, start_time
                )
            else:
                value = self._minimax_simple(game_copy, depth - 1, False, start_time)
            
            # 更新历史启发
            self._update_history_heuristic(action, value)
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action, best_value
    
    def _minimax_alpha_beta(self, game, depth: int, alpha: float, beta: float, 
                           maximizing: bool, start_time: float) -> float:
        """
        增强的Minimax算法 with Alpha-Beta剪枝
        """
        # 时间检查
        if time.time() - start_time > self.evaluation_timeout:
            return self._evaluate_game_state(game)
        
        # 置换表查找
        game_hash = self._get_game_hash(game)
        if self.use_transposition_table and game_hash in self.evaluation_cache:
            cache_entry = self.evaluation_cache[game_hash]
            if cache_entry['depth'] >= depth:
                self.cache_hits += 1
                return cache_entry['value']
        
        self.nodes_evaluated += 1
        
        # 终止条件
        if depth == 0 or game.is_terminal():
            value = self._evaluate_game_state(game)
            if self.use_transposition_table:
                self.evaluation_cache[game_hash] = {'value': value, 'depth': depth}
            return value
        
        # 获取并排序动作
        valid_actions = game.get_valid_actions()
        if not valid_actions:
            value = self._evaluate_game_state(game)
            if self.use_transposition_table:
                self.evaluation_cache[game_hash] = {'value': value, 'depth': depth}
            return value
        
        # 动作排序优化（包括杀手启发和历史启发）
        valid_actions = self._sort_actions_advanced(valid_actions, game, depth)
        
        if maximizing:
            max_value = float('-inf')
            for action in valid_actions:
                game_copy = game.clone()
                game_copy.step(action)
                
                value = self._minimax_alpha_beta(
                    game_copy, depth - 1, alpha, beta, False, start_time
                )
                
                if value > max_value:
                    max_value = value
                    # 更新杀手移动
                    self._update_killer_move(depth, action, value)
                
                alpha = max(alpha, value)
                
                # Beta剪枝
                if beta <= alpha:
                    self.pruning_count += 1
                    break
            
            if self.use_transposition_table:
                self.evaluation_cache[game_hash] = {'value': max_value, 'depth': depth}
            return max_value
        else:
            min_value = float('inf')
            for action in valid_actions:
                game_copy = game.clone()
                game_copy.step(action)
                
                value = self._minimax_alpha_beta(
                    game_copy, depth - 1, alpha, beta, True, start_time
                )
                
                if value < min_value:
                    min_value = value
                    self._update_killer_move(depth, action, value)
                
                beta = min(beta, value)
                
                # Alpha剪枝
                if beta <= alpha:
                    self.pruning_count += 1
                    break
            
            if self.use_transposition_table:
                self.evaluation_cache[game_hash] = {'value': min_value, 'depth': depth}
            return min_value
    
    def _minimax_simple(self, game, depth: int, maximizing: bool, start_time: float) -> float:
        """简单Minimax算法（不使用剪枝）"""
        if time.time() - start_time > self.evaluation_timeout:
            return self._evaluate_game_state(game)
        
        self.nodes_evaluated += 1
        
        if depth == 0 or game.is_terminal():
            return self._evaluate_game_state(game)
        
        valid_actions = game.get_valid_actions()
        if not valid_actions:
            return self._evaluate_game_state(game)
        
        if maximizing:
            max_value = float('-inf')
            for action in valid_actions:
                game_copy = game.clone()
                game_copy.step(action)
                value = self._minimax_simple(game_copy, depth - 1, False, start_time)
                max_value = max(max_value, value)
            return max_value
        else:
            min_value = float('inf')
            for action in valid_actions:
                game_copy = game.clone()
                game_copy.step(action)
                value = self._minimax_simple(game_copy, depth - 1, True, start_time)
                min_value = min(min_value, value)
            return min_value
    
    def _evaluate_game_state(self, game) -> float:
        """
        增强的游戏状态评估函数
        """
        # 检查游戏是否结束
        if game.is_terminal():
            winner = game.get_winner()
            if winner == self.player_id:
                return 10000.0  # 获胜
            elif winner is not None:
                return -10000.0  # 失败
            else:
                return 0.0  # 平局
        
        # 根据游戏类型进行评估
        if hasattr(game, 'board'):
            return self._evaluate_gomoku_state_enhanced(game)
        elif hasattr(game, 'snake1'):
            return self._evaluate_snake_state_enhanced(game)
        elif hasattr(game, 'grid1'):
            return self._evaluate_sokoban_state_enhanced(game)
        else:
            return random.uniform(-1, 1)
    
    def _evaluate_gomoku_state_enhanced(self, game) -> float:
        """增强的五子棋状态评估"""
        board = game.board
        score = 0.0
        
        # 评估模式分数表
        pattern_scores = {
            5: 100000,   # 五连
            4: 10000,    # 活四
            3: 1000,     # 活三
            2: 100,      # 活二
            1: 10        # 单子
        }
        
        # 阻塞模式分数（对手的威胁）
        block_scores = {
            4: 50000,    # 阻止对手活四
            3: 5000,     # 阻止对手活三
            2: 500       # 阻止对手活二
        }
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i, j] != 0:
                    player = board[i, j]
                    multiplier = 1 if player == self.player_id else -1
                    
                    for dr, dc in directions:
                        # 计算连续子数和模式
                        pattern_info = self._analyze_pattern(board, i, j, dr, dc, player)
                        consecutive = pattern_info['consecutive']
                        is_blocked = pattern_info['blocked']
                        
                        if consecutive > 0:
                            base_score = pattern_scores.get(consecutive, 1)
                            if is_blocked:
                                base_score = base_score // 2  # 被阻塞的模式分数减半
                            
                            score += base_score * multiplier
                            
                            # 对手威胁评估
                            if player != self.player_id and consecutive >= 2:
                                block_score = block_scores.get(consecutive, 0)
                                score += block_score  # 我们需要阻止对手
        
        # 位置价值评估（更精细）
        score += self._evaluate_position_value(board)
        
        # 控制中心奖励
        score += self._evaluate_center_control(board)
        
        return score
    
    def _analyze_pattern(self, board: np.ndarray, row: int, col: int, 
                        dr: int, dc: int, player: int) -> Dict[str, Any]:
        """分析棋子模式"""
        consecutive = 1
        blocked_start = False
        blocked_end = False
        
        # 正方向计数
        r, c = row + dr, col + dc
        while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
               board[r, c] == player):
            consecutive += 1
            r += dr
            c += dc
        
        # 检查正方向是否被阻塞
        if (r < 0 or r >= board.shape[0] or c < 0 or c >= board.shape[1] or 
            board[r, c] != 0):
            blocked_end = True
        
        # 负方向计数
        r, c = row - dr, col - dc
        while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
               board[r, c] == player):
            consecutive += 1
            r -= dr
            c -= dc
        
        # 检查负方向是否被阻塞
        if (r < 0 or r >= board.shape[0] or c < 0 or c >= board.shape[1] or 
            board[r, c] != 0):
            blocked_start = True
        
        return {
            'consecutive': consecutive,
            'blocked': blocked_start and blocked_end,
            'semi_blocked': blocked_start or blocked_end
        }
    
    def _evaluate_position_value(self, board: np.ndarray) -> float:
        """评估位置价值"""
        score = 0.0
        center = board.shape[0] // 2
        
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i, j] == self.player_id:
                    # 距离中心越近价值越高
                    distance_to_center = max(abs(i - center), abs(j - center))
                    position_value = max(0, 7 - distance_to_center)
                    score += position_value
                elif board[i, j] != 0:
                    # 对手棋子
                    distance_to_center = max(abs(i - center), abs(j - center))
                    position_value = max(0, 7 - distance_to_center)
                    score -= position_value * 0.5
        
        return score
    
    def _evaluate_center_control(self, board: np.ndarray) -> float:
        """评估中心控制"""
        center = board.shape[0] // 2
        score = 0.0
        
        # 检查中心9x9区域的控制
        for i in range(max(0, center-4), min(board.shape[0], center+5)):
            for j in range(max(0, center-4), min(board.shape[1], center+5)):
                if board[i, j] == self.player_id:
                    score += 2
                elif board[i, j] != 0:
                    score -= 1
        
        return score
    
    def _evaluate_snake_state_enhanced(self, game) -> float:
        """增强的贪吃蛇状态评估"""
        score = 0.0
        
        # 获取蛇的信息
        if self.player_id == 1:
            my_snake = game.snake1
            opponent_snake = game.snake2
            my_alive = game.alive1
            opponent_alive = game.alive2
        else:
            my_snake = game.snake2
            opponent_snake = game.snake1
            my_alive = game.alive2
            opponent_alive = game.alive1
        
        # 生存状态评估
        if not my_alive:
            return -10000.0
        if not opponent_alive:
            return 10000.0
        
        if not my_snake or not opponent_snake:
            return 0.0
        
        # 长度优势
        length_diff = len(my_snake) - len(opponent_snake)
        score += length_diff * 100
        
        my_head = my_snake[0]
        opponent_head = opponent_snake[0]
        
        # 食物控制评估
        if game.foods:
            for food in game.foods:
                my_distance = abs(my_head[0] - food[0]) + abs(my_head[1] - food[1])
                opponent_distance = abs(opponent_head[0] - food[0]) + abs(opponent_head[1] - food[1])
                
                # 距离食物更近的奖励
                if my_distance < opponent_distance:
                    score += 50
                elif my_distance > opponent_distance:
                    score -= 30
                
                # 食物距离奖励
                score += 200 / (my_distance + 1)
        
        # 空间控制评估
        my_accessible_space = self._calculate_accessible_space(my_head, game, my_snake + opponent_snake)
        opponent_accessible_space = self._calculate_accessible_space(opponent_head, game, my_snake + opponent_snake)
        
        space_advantage = my_accessible_space - opponent_accessible_space
        score += space_advantage * 2
        
        # 安全性评估
        my_safety = self._evaluate_position_safety(my_head, game)
        opponent_safety = self._evaluate_position_safety(opponent_head, game)
        score += (my_safety - opponent_safety) * 10
        
        # 中心位置奖励
        center = game.board_size // 2
        my_center_distance = abs(my_head[0] - center) + abs(my_head[1] - center)
        score += 20 / (my_center_distance + 1)
        
        return score
    
    def _evaluate_sokoban_state_enhanced(self, game) -> float:
        """增强的推箱子状态评估"""
        score = 0.0
        
        if self.player_id == 1:
            boxes_in_place = game.boxes_in_place1
            moves = game.moves1
            player_pos = game.player1_pos
        else:
            boxes_in_place = game.boxes_in_place2
            moves = game.moves2
            player_pos = game.player2_pos
        
        target_boxes = game.original_level['target_boxes']
        
        # 完成度评估
        completion_rate = boxes_in_place / target_boxes
        score += completion_rate * 1000
        
        # 移动效率奖励
        if moves > 0:
            efficiency = boxes_in_place / moves
            score += efficiency * 100
        
        # 剩余箱子距离目标的总距离（越小越好）
        total_distance = self._calculate_total_box_distance(game)
        score -= total_distance * 5
        
        # 玩家位置评估
        score += self._evaluate_player_position(game, player_pos)
        
        return score
    
    def _calculate_accessible_space(self, start: Tuple[int, int], game: Any, 
                                   obstacles: List[Tuple[int, int]]) -> int:
        """计算可达空间"""
        visited = set()
        queue = [start]
        visited.add(start)
        
        while queue:
            current = queue.pop(0)
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (current[0] + dr, current[1] + dc)
                
                if (0 <= new_pos[0] < game.board_size and 
                    0 <= new_pos[1] < game.board_size and
                    new_pos not in visited and
                    new_pos not in obstacles):
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        return len(visited)
    
    def _evaluate_position_safety(self, position: Tuple[int, int], game: Any) -> float:
        """评估位置安全性"""
        safety_score = 0.0
        
        # 检查周围的自由度
        free_directions = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (position[0] + dr, position[1] + dc)
            if (0 <= new_pos[0] < game.board_size and 
                0 <= new_pos[1] < game.board_size and
                new_pos not in game.snake1 and 
                new_pos not in game.snake2):
                free_directions += 1
        
        safety_score += free_directions * 5
        
        # 距离边界的安全距离
        border_distance = min(
            position[0], 
            position[1], 
            game.board_size - 1 - position[0], 
            game.board_size - 1 - position[1]
        )
        safety_score += border_distance
        
        return safety_score
    
    def _order_actions(self, actions: List[Any], env: Any) -> List[Any]:
        """基础动作排序"""
        if hasattr(env, 'game') and hasattr(env.game, 'board'):
            # 五子棋：优先中心位置
            center = env.game.board.shape[0] // 2
            actions.sort(key=lambda a: abs(a[0] - center) + abs(a[1] - center))
        
        return actions
    
    def _sort_actions_advanced(self, actions: List[Any], game, depth: int) -> List[Any]:
        """高级动作排序（包括杀手启发和历史启发）"""
        scored_actions = []
        
        for action in actions:
            score = 0
            
            # 杀手启发
            if depth in self.killer_moves and action in self.killer_moves[depth]:
                score += 1000
            
            # 历史启发
            if action in self.history_heuristic:
                score += self.history_heuristic[action]
            
            # 基础位置评估
            if hasattr(game, 'board'):
                if isinstance(action, tuple) and len(action) == 2:
                    center = game.board.shape[0] // 2
                    distance = abs(action[0] - center) + abs(action[1] - center)
                    score += max(0, 10 - distance)
            
            scored_actions.append((action, score))
        
        # 按分数降序排序
        scored_actions.sort(key=lambda x: x[1], reverse=True)
        return [action for action, _ in scored_actions]
    
    def _update_killer_move(self, depth: int, action: Any, value: float):
        """更新杀手移动"""
        if depth not in self.killer_moves:
            self.killer_moves[depth] = []
        
        if action not in self.killer_moves[depth]:
            self.killer_moves[depth].append(action)
            # 限制杀手移动数量
            if len(self.killer_moves[depth]) > 2:
                self.killer_moves[depth].pop(0)
    
    def _update_history_heuristic(self, action: Any, value: float):
        """更新历史启发"""
        if action not in self.history_heuristic:
            self.history_heuristic[action] = 0
        
        self.history_heuristic[action] += max(0, value)
        
        # 定期衰减历史值
        if len(self.history_heuristic) > 1000:
            for key in self.history_heuristic:
                self.history_heuristic[key] *= 0.9
    
    def _safe_clone(self, env: Any) -> Any:
        """安全的环境克隆"""
        try:
            if hasattr(env, 'clone'):
                return env.clone()
            elif hasattr(env, 'game') and hasattr(env.game, 'clone'):
                cloned_env = type(env)(env.board_size if hasattr(env, 'board_size') else 15)
                cloned_env.game = env.game.clone()
                return cloned_env
            else:
                return None
        except Exception as e:
            return None
    
    def _get_game_hash(self, game) -> str:
        """获取游戏状态的哈希值"""
        try:
            if hasattr(game, 'board'):
                return f"board_{hash(game.board.tobytes())}_{game.current_player}"
            elif hasattr(game, 'snake1'):
                return f"snake_{hash(str(game.snake1))}_{hash(str(game.snake2))}_{hash(str(game.foods))}"
            else:
                return str(hash(str(game.get_state())))
        except:
            return str(random.randint(0, 1000000))
    
    def reset(self):
        """重置Minimax Bot"""
        super().reset()
        self.evaluation_cache.clear()
        self.killer_moves.clear()
        self.history_heuristic.clear()
        self.nodes_evaluated = 0
        self.pruning_count = 0
        self.cache_hits = 0
    
    def get_info(self) -> Dict[str, Any]:
        """获取Minimax Bot信息"""
        info = super().get_info()
        info.update({
            'type': 'Enhanced Minimax',
            'description': '增强的Alpha-Beta剪枝Minimax算法',
            'strategy': f'Enhanced Minimax depth={self.max_depth}, iterative={self.use_iterative_deepening}',
            'max_depth': self.max_depth,
            'use_alpha_beta': self.use_alpha_beta,
            'use_iterative_deepening': self.use_iterative_deepening,
            'time_limit': self.evaluation_timeout,
            'nodes_evaluated': self.nodes_evaluated,
            'pruning_count': self.pruning_count,
            'cache_hits': self.cache_hits,
            'cache_size': len(self.evaluation_cache),
            'killer_moves': len(self.killer_moves),
            'history_entries': len(self.history_heuristic)
        })
        return info
