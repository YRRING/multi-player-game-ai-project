"""
贪吃蛇专用AI智能体 - 完善优化版本
包含高级路径规划、对手预测和生存策略
"""

import random
import time
import heapq
from typing import List, Tuple, Any, Optional, Dict, Set
from collections import deque
import numpy as np
from agents.base_agent import BaseAgent


class SnakeAI(BaseAgent):
    """基础贪吃蛇AI智能体 - 优化版本"""
    
    def __init__(self, name: str = "SnakeAI", player_id: int = 1):
        super().__init__(name, player_id)
        self.safety_threshold = 0.6
        self.food_priority = 1.0
        self.survival_priority = 2.0
        self.aggression_level = 0.3
        
        # 路径缓存
        self.path_cache = {}
        self.cache_ttl = 5  # 缓存存活时间
        
        # 决策历史
        self.decision_history = []
        self.max_history = 20
        
        # 性能统计
        self.pathfinding_time = 0
        self.safety_checks = 0
        
    def get_action(self, observation: Any, env: Any) -> Tuple[int, int]:
        """获取优化的动作"""
        start_time = time.time()
        
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return (0, 1)  # 默认向右
        
        # 获取当前蛇的信息
        game = env.game
        if self.player_id == 1:
            snake = game.snake1
            opponent_snake = game.snake2
            current_direction = game.direction1
            alive = game.alive1
        else:
            snake = game.snake2
            opponent_snake = game.snake1
            current_direction = game.direction2
            alive = game.alive2
        
        if not snake or not alive:
            return random.choice(valid_actions)
        
        head = snake[0]
        
        # 策略决策树
        action = self._strategic_decision(head, snake, opponent_snake, game, valid_actions, current_direction)
        
        # 记录决策
        self._record_decision(action, head, game)
        
        # 更新统计
        move_time = time.time() - start_time
        self.total_moves += 1
        self.total_time += move_time
        
        return action
    
    def _strategic_decision(self, head: Tuple[int, int], snake: List[Tuple[int, int]], 
                          opponent_snake: List[Tuple[int, int]], game: Any, 
                          valid_actions: List[Tuple[int, int]], 
                          current_direction: Tuple[int, int]) -> Tuple[int, int]:
        """战略决策核心"""
        
        # 1. 紧急避险
        emergency_action = self._emergency_avoidance(head, snake, opponent_snake, game, valid_actions)
        if emergency_action:
            return emergency_action
        
        # 2. 获取安全动作
        safe_actions = self._get_safe_actions_advanced(head, snake, opponent_snake, game, valid_actions)
        
        if not safe_actions:
            # 如果没有安全动作，选择最不危险的
            return self._choose_least_dangerous_action(head, game, valid_actions)
        
        # 3. 食物获取策略
        if game.foods:
            food_action = self._food_acquisition_strategy(head, snake, opponent_snake, game, safe_actions)
            if food_action:
                return food_action
        
        # 4. 领土控制策略
        territory_action = self._territory_control_strategy(head, snake, opponent_snake, game, safe_actions)
        if territory_action:
            return territory_action
        
        # 5. 生存策略（保持在安全区域）
        survival_action = self._survival_strategy(head, snake, game, safe_actions)
        if survival_action:
            return survival_action
        
        # 6. 默认选择
        return random.choice(safe_actions)
    
    def _emergency_avoidance(self, head: Tuple[int, int], snake: List[Tuple[int, int]], 
                           opponent_snake: List[Tuple[int, int]], game: Any, 
                           valid_actions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """紧急避险策略"""
        
        # 检查是否即将发生碰撞
        immediate_threats = []
        
        # 检查对手头部威胁
        if opponent_snake:
            opponent_head = opponent_snake[0]
            head_distance = abs(head[0] - opponent_head[0]) + abs(head[1] - opponent_head[1])
            
            if head_distance <= 2:  # 对手很近
                # 预测对手可能的移动
                opponent_possible_moves = self._predict_opponent_moves(opponent_head, game)
                
                for action in valid_actions:
                    new_head = (head[0] + action[0], head[1] + action[1])
                    if new_head in opponent_possible_moves:
                        continue  # 跳过可能与对手碰撞的动作
                    
                    if self._is_position_safe_immediate(new_head, snake, opponent_snake, game):
                        return action
        
        # 检查空间陷阱
        for action in valid_actions:
            new_head = (head[0] + action[0], head[1] + action[1])
            accessible_space = self._quick_space_check(new_head, snake, opponent_snake, game)
            
            if accessible_space >= len(snake) * 0.8:  # 有足够空间
                return action
        
        return None
    
    def _get_safe_actions_advanced(self, head: Tuple[int, int], snake: List[Tuple[int, int]], 
                                 opponent_snake: List[Tuple[int, int]], game: Any, 
                                 valid_actions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """获取高级安全动作"""
        safe_actions = []
        
        for action in valid_actions:
            new_head = (head[0] + action[0], head[1] + action[1])
            
            # 基本安全检查
            if not self._is_position_safe_immediate(new_head, snake, opponent_snake, game):
                continue
            
            # 预测性安全检查
            safety_score = self._calculate_position_safety_score(new_head, snake, opponent_snake, game)
            
            if safety_score >= self.safety_threshold:
                safe_actions.append(action)
        
        return safe_actions
    
    def _food_acquisition_strategy(self, head: Tuple[int, int], snake: List[Tuple[int, int]], 
                                 opponent_snake: List[Tuple[int, int]], game: Any, 
                                 safe_actions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """食物获取策略"""
        if not game.foods:
            return None
        
        # 评估每个食物的价值
        food_evaluations = []
        
        for food in game.foods:
            evaluation = self._evaluate_food_value(head, food, snake, opponent_snake, game)
            food_evaluations.append((food, evaluation))
        
        # 按价值排序
        food_evaluations.sort(key=lambda x: x[1], reverse=True)
        
        # 尝试获取最有价值的食物
        for food, value in food_evaluations:
            if value <= 0:
                continue
            
            # 使用A*寻找路径
            path = self._a_star_pathfinding_enhanced(head, food, snake, opponent_snake, game)
            
            if path and len(path) > 1:
                next_pos = path[1]
                action = (next_pos[0] - head[0], next_pos[1] - head[1])
                
                if action in safe_actions:
                    return action
            
            # 如果A*失败，使用贪心策略
            greedy_action = self._greedy_move_to_food(head, food, safe_actions)
            if greedy_action:
                return greedy_action
        
        return None
    
    def _territory_control_strategy(self, head: Tuple[int, int], snake: List[Tuple[int, int]], 
                                  opponent_snake: List[Tuple[int, int]], game: Any, 
                                  safe_actions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """领土控制策略"""
        if not opponent_snake or len(snake) < 5:  # 只有当蛇足够长时才考虑领土控制
            return None
        
        opponent_head = opponent_snake[0]
        
        # 计算控制区域的价值
        best_action = None
        max_control_value = 0
        
        for action in safe_actions:
            new_head = (head[0] + action[0], head[1] + action[1])
            
            # 计算这个位置的控制价值
            control_value = self._calculate_territory_control_value(new_head, opponent_head, game)
            
            # 考虑食物控制
            food_control_bonus = self._calculate_food_control_bonus(new_head, opponent_head, game)
            
            total_value = control_value + food_control_bonus
            
            if total_value > max_control_value:
                max_control_value = total_value
                best_action = action
        
        # 只有当控制价值足够高时才执行
        if max_control_value > 10:
            return best_action
        
        return None
    
    def _survival_strategy(self, head: Tuple[int, int], snake: List[Tuple[int, int]], 
                         game: Any, safe_actions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """生存策略"""
        best_action = None
        max_survival_score = 0
        
        for action in safe_actions:
            new_head = (head[0] + action[0], head[1] + action[1])
            
            # 计算生存分数
            survival_score = self._calculate_survival_score(new_head, snake, game)
            
            if survival_score > max_survival_score:
                max_survival_score = survival_score
                best_action = action
        
        return best_action
    
    def _evaluate_food_value(self, head: Tuple[int, int], food: Tuple[int, int], 
                           snake: List[Tuple[int, int]], opponent_snake: List[Tuple[int, int]], 
                           game: Any) -> float:
        """评估食物价值"""
        value = 100.0  # 基础价值
        
        # 距离因子
        my_distance = self._manhattan_distance(head, food)
        value += 50 / (my_distance + 1)
        
        # 竞争因子
        if opponent_snake:
            opponent_head = opponent_snake[0]
            opponent_distance = self._manhattan_distance(opponent_head, food)
            
            if my_distance < opponent_distance:
                value += 30  # 我们更近
            elif my_distance > opponent_distance:
                value -= 20  # 对手更近
        
        # 安全因子
        path_safety = self._evaluate_path_safety(head, food, snake, opponent_snake, game)
        value += path_safety
        
        # 位置因子（避免边角食物，除非很安全）
        border_distance = min(food[0], food[1], game.board_size - 1 - food[0], game.board_size - 1 - food[1])
        if border_distance < 2:
            value -= 15
        
        return value
    
    def _a_star_pathfinding_enhanced(self, start: Tuple[int, int], goal: Tuple[int, int], 
                                   snake: List[Tuple[int, int]], opponent_snake: List[Tuple[int, int]], 
                                   game: Any) -> Optional[List[Tuple[int, int]]]:
        """增强的A*寻路算法"""
        
        # 检查缓存
        cache_key = (start, goal, len(snake), len(opponent_snake) if opponent_snake else 0)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        pathfind_start = time.time()
        
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (pos[0] + dr, pos[1] + dc)
                if self._is_valid_pathfinding_position(new_pos, snake, opponent_snake, game):
                    neighbors.append(new_pos)
            return neighbors
        
        def get_cost(current: Tuple[int, int], neighbor: Tuple[int, int]) -> float:
            base_cost = 1.0
            
            # 增加危险区域的成本
            danger_cost = self._calculate_position_danger(neighbor, snake, opponent_snake, game)
            
            # 边界惩罚
            border_distance = min(neighbor[0], neighbor[1], 
                                game.board_size - 1 - neighbor[0], 
                                game.board_size - 1 - neighbor[1])
            if border_distance < 2:
                base_cost += 0.5
            
            return base_cost + danger_cost
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        visited = set()
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == goal:
                # 重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                result_path = path[::-1]
                
                # 缓存结果
                self.path_cache[cache_key] = result_path
                self.pathfinding_time += time.time() - pathfind_start
                
                return result_path
            
            for neighbor in get_neighbors(current):
                if neighbor in visited:
                    continue
                
                tentative_g = g_score[current] + get_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        self.pathfinding_time += time.time() - pathfind_start
        return None
    
    def _predict_opponent_moves(self, opponent_head: Tuple[int, int], game: Any) -> List[Tuple[int, int]]:
        """预测对手可能的移动"""
        predictions = []
        
        # 基于食物的预测
        if game.foods:
            nearest_food = min(game.foods, 
                             key=lambda f: self._manhattan_distance(opponent_head, f))
            
            # 预测对手朝最近食物移动
            dx = nearest_food[0] - opponent_head[0]
            dy = nearest_food[1] - opponent_head[1]
            
            if abs(dx) > abs(dy):
                if dx > 0:
                    predictions.append((opponent_head[0] + 1, opponent_head[1]))
                else:
                    predictions.append((opponent_head[0] - 1, opponent_head[1]))
            else:
                if dy > 0:
                    predictions.append((opponent_head[0], opponent_head[1] + 1))
                else:
                    predictions.append((opponent_head[0], opponent_head[1] - 1))
        
        # 添加所有可能的移动
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (opponent_head[0] + dr, opponent_head[1] + dc)
            if self._is_valid_position_basic(new_pos, game):
                predictions.append(new_pos)
        
        return predictions
    
    def _calculate_position_safety_score(self, position: Tuple[int, int], 
                                       snake: List[Tuple[int, int]], 
                                       opponent_snake: List[Tuple[int, int]], 
                                       game: Any) -> float:
        """计算位置安全分数"""
        self.safety_checks += 1
        score = 1.0
        
        # 1. 可达空间
        accessible_space = self._calculate_accessible_space_limited(position, snake, opponent_snake, game)
        required_space = len(snake) * 0.6
        
        if accessible_space < required_space:
            score -= 0.5
        
        # 2. 边界距离
        border_distance = min(position[0], position[1], 
                            game.board_size - 1 - position[0], 
                            game.board_size - 1 - position[1])
        if border_distance < 2:
            score -= 0.2
        
        # 3. 对手威胁
        if opponent_snake:
            opponent_head = opponent_snake[0]
            threat_distance = self._manhattan_distance(position, opponent_head)
            if threat_distance <= 2:
                score -= 0.3
        
        # 4. 逃生路线
        escape_routes = self._count_escape_routes(position, snake, opponent_snake, game)
        if escape_routes < 2:
            score -= 0.2
        
        return max(0.0, score)
    
    def _calculate_territory_control_value(self, position: Tuple[int, int], 
                                         opponent_head: Tuple[int, int], 
                                         game: Any) -> float:
        """计算领土控制价值"""
        my_distance_to_center = self._manhattan_distance(position, (game.board_size // 2, game.board_size // 2))
        opponent_distance_to_center = self._manhattan_distance(opponent_head, (game.board_size // 2, game.board_size // 2))
        
        control_value = 0.0
        
        # 中心控制奖励
        if my_distance_to_center < opponent_distance_to_center:
            control_value += 5.0
        
        # 空间分割
        controlled_space = self._calculate_controlled_space_simple(position, opponent_head, game)
        control_value += controlled_space * 0.1
        
        return control_value
    
    def _calculate_food_control_bonus(self, position: Tuple[int, int], 
                                    opponent_head: Tuple[int, int], 
                                    game: Any) -> float:
        """计算食物控制奖励"""
        if not game.foods:
            return 0.0
        
        bonus = 0.0
        
        for food in game.foods:
            my_distance = self._manhattan_distance(position, food)
            opponent_distance = self._manhattan_distance(opponent_head, food)
            
            if my_distance < opponent_distance:
                bonus += 3.0
            elif my_distance == opponent_distance:
                bonus += 1.0
        
        return bonus
    
    def _calculate_survival_score(self, position: Tuple[int, int], 
                                snake: List[Tuple[int, int]], 
                                game: Any) -> float:
        """计算生存分数"""
        score = 0.0
        
        # 可达空间
        accessible_space = self._calculate_accessible_space_limited(position, snake, [], game)
        score += accessible_space * 0.5
        
        # 自由度
        free_directions = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (position[0] + dr, position[1] + dc)
            if self._is_valid_position_basic(new_pos, game):
                free_directions += 1
        
        score += free_directions * 2
        
        # 中心位置奖励
        center_distance = self._manhattan_distance(position, (game.board_size // 2, game.board_size // 2))
        score += max(0, 10 - center_distance)
        
        return score
    
    def _quick_space_check(self, position: Tuple[int, int], 
                         snake: List[Tuple[int, int]], 
                         opponent_snake: List[Tuple[int, int]], 
                         game: Any) -> int:
        """快速空间检查（限制搜索深度）"""
        visited = set()
        queue = deque([position])
        visited.add(position)
        
        obstacles = set(snake + (opponent_snake or []))
        
        while queue and len(visited) < 50:  # 限制搜索范围
            current = queue.popleft()
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (current[0] + dr, current[1] + dc)
                
                if (new_pos not in visited and 
                    self._is_valid_position_basic(new_pos, game) and
                    new_pos not in obstacles):
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        return len(visited)
    
    def _calculate_accessible_space_limited(self, position: Tuple[int, int], 
                                          snake: List[Tuple[int, int]], 
                                          opponent_snake: List[Tuple[int, int]], 
                                          game: Any) -> int:
        """计算有限的可达空间（优化性能）"""
        visited = set()
        queue = deque([position])
        visited.add(position)
        
        obstacles = set(snake[:-1] + (opponent_snake[:-1] if opponent_snake else []))
        max_search = min(100, game.board_size * game.board_size // 4)
        
        while queue and len(visited) < max_search:
            current = queue.popleft()
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (current[0] + dr, current[1] + dc)
                
                if (new_pos not in visited and 
                    0 <= new_pos[0] < game.board_size and 
                    0 <= new_pos[1] < game.board_size and
                    new_pos not in obstacles):
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        return len(visited)
    
    def _count_escape_routes(self, position: Tuple[int, int], 
                           snake: List[Tuple[int, int]], 
                           opponent_snake: List[Tuple[int, int]], 
                           game: Any) -> int:
        """计算逃生路线数量"""
        escape_count = 0
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            direction_pos = (position[0] + dr, position[1] + dc)
            
            if self._is_valid_pathfinding_position(direction_pos, snake, opponent_snake, game):
                # 检查这个方向是否有足够的空间
                space_in_direction = self._quick_space_check(direction_pos, snake, opponent_snake, game)
                if space_in_direction >= 10:  # 至少有10个空格的逃生空间
                    escape_count += 1
        
        return escape_count
    
    def _calculate_controlled_space_simple(self, my_pos: Tuple[int, int], 
                                         opponent_pos: Tuple[int, int], 
                                         game: Any) -> int:
        """简单的控制空间计算"""
        controlled = 0
        
        # 简化版本：只检查一小块区域
        center = game.board_size // 2
        for i in range(max(0, center - 5), min(game.board_size, center + 6)):
            for j in range(max(0, center - 5), min(game.board_size, center + 6)):
                pos = (i, j)
                my_distance = self._manhattan_distance(my_pos, pos)
                opponent_distance = self._manhattan_distance(opponent_pos, pos)
                
                if my_distance < opponent_distance:
                    controlled += 1
        
        return controlled
    
    def _is_position_safe_immediate(self, position: Tuple[int, int], 
                                  snake: List[Tuple[int, int]], 
                                  opponent_snake: List[Tuple[int, int]], 
                                  game: Any) -> bool:
        """检查位置是否立即安全"""
        row, col = position
        
        # 边界检查
        if (row < 0 or row >= game.board_size or 
            col < 0 or col >= game.board_size):
            return False
        
        # 蛇身检查
        if position in snake[:-1]:  # 排除尾巴
            return False
        
        if opponent_snake and position in opponent_snake[:-1]:
            return False
        
        return True
    
    def _is_valid_pathfinding_position(self, position: Tuple[int, int], 
                                     snake: List[Tuple[int, int]], 
                                     opponent_snake: List[Tuple[int, int]], 
                                     game: Any) -> bool:
        """检查位置是否适合寻路"""
        row, col = position
        
        if (row < 0 or row >= game.board_size or 
            col < 0 or col >= game.board_size):
            return False
        
        # 寻路时考虑整个蛇身
        if position in snake or (opponent_snake and position in opponent_snake):
            return False
        
        return True
    
    def _is_valid_position_basic(self, position: Tuple[int, int], game: Any) -> bool:
        """基本位置有效性检查"""
        row, col = position
        return (0 <= row < game.board_size and 0 <= col < game.board_size)
    
    def _calculate_position_danger(self, position: Tuple[int, int], 
                                 snake: List[Tuple[int, int]], 
                                 opponent_snake: List[Tuple[int, int]], 
                                 game: Any) -> float:
        """计算位置危险度"""
        danger = 0.0
        
        # 边界危险
        border_distance = min(position[0], position[1], 
                            game.board_size - 1 - position[0], 
                            game.board_size - 1 - position[1])
        if border_distance < 2:
            danger += 0.3
        
        # 对手威胁
        if opponent_snake:
            opponent_head = opponent_snake[0]
            distance_to_opponent = self._manhattan_distance(position, opponent_head)
            if distance_to_opponent <= 3:
                danger += (4 - distance_to_opponent) * 0.1
        
        return danger
    
    def _evaluate_path_safety(self, start: Tuple[int, int], goal: Tuple[int, int], 
                            snake: List[Tuple[int, int]], opponent_snake: List[Tuple[int, int]], 
                            game: Any) -> float:
        """评估路径安全性"""
        # 简单检查：直线路径上是否有障碍
        safety_score = 10.0
        
        dx = 1 if goal[0] > start[0] else -1 if goal[0] < start[0] else 0
        dy = 1 if goal[1] > start[1] else -1 if goal[1] < start[1] else 0
        
        current = start
        while current != goal:
            if dx != 0:
                current = (current[0] + dx, current[1])
            elif dy != 0:
                current = (current[0], current[1] + dy)
            else:
                break
            
            if not self._is_valid_pathfinding_position(current, snake, opponent_snake, game):
                safety_score -= 2.0
            
            if safety_score <= 0:
                break
        
        return max(0.0, safety_score)
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """计算曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _greedy_move_to_food(self, head: Tuple[int, int], target_food: Tuple[int, int], 
                           safe_actions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """贪心移动到食物"""
        dx = target_food[0] - head[0]
        dy = target_food[1] - head[1]
        
        preferred_actions = []
        
        if abs(dx) > abs(dy):
            if dx > 0:
                preferred_actions.append((1, 0))
            elif dx < 0:
                preferred_actions.append((-1, 0))
            
            if dy > 0:
                preferred_actions.append((0, 1))
            elif dy < 0:
                preferred_actions.append((0, -1))
        else:
            if dy > 0:
                preferred_actions.append((0, 1))
            elif dy < 0:
                preferred_actions.append((0, -1))
            
            if dx > 0:
                preferred_actions.append((1, 0))
            elif dx < 0:
                preferred_actions.append((-1, 0))
        
        for action in preferred_actions:
            if action in safe_actions:
                return action
        
        return None
    
    def _choose_least_dangerous_action(self, head: Tuple[int, int], game: Any, 
                                     valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """选择最不危险的动作"""
        scored_actions = []
        
        for action in valid_actions:
            new_head = (head[0] + action[0], head[1] + action[1])
            
            # 计算危险分数（越低越好）
            danger_score = 0.0
            
            # 边界惩罚
            if (new_head[0] <= 0 or new_head[0] >= game.board_size - 1 or
                new_head[1] <= 0 or new_head[1] >= game.board_size - 1):
                danger_score += 10.0
            
            # 估算存活时间
            survival_time = self._estimate_survival_time(new_head, game)
            danger_score -= survival_time
            
            scored_actions.append((action, danger_score))
        
        # 选择危险分数最低的动作
        scored_actions.sort(key=lambda x: x[1])
        return scored_actions[0][0]
    
    def _estimate_survival_time(self, position: Tuple[int, int], game: Any) -> int:
        """估算生存时间"""
        if not self._is_valid_position_basic(position, game):
            return 0
        
        # 简单估算：可达空间大小
        accessible_space = self._quick_space_check(position, [], [], game)
        return min(accessible_space, 50)
    
    def _record_decision(self, action: Tuple[int, int], head: Tuple[int, int], game: Any):
        """记录决策历史"""
        decision = {
            'action': action,
            'head_position': head,
            'food_count': len(game.foods),
            'timestamp': time.time()
        }
        
        self.decision_history.append(decision)
        
        # 保持历史长度
        if len(self.decision_history) > self.max_history:
            self.decision_history.pop(0)
    
    def reset(self):
        """重置SnakeAI"""
        super().reset()
        self.path_cache.clear()
        self.decision_history.clear()
        self.pathfinding_time = 0
        self.safety_checks = 0
    
    def get_info(self) -> Dict[str, Any]:
        """获取SnakeAI信息"""
        info = super().get_info()
        info.update({
            'type': 'Enhanced Snake AI',
            'description': '增强的贪吃蛇AI，包含高级路径规划和生存策略',
            'strategy': 'Advanced pathfinding + survival strategy',
            'safety_threshold': self.safety_threshold,
            'aggression_level': self.aggression_level,
            'pathfinding_time': self.pathfinding_time,
            'safety_checks': self.safety_checks,
            'cache_size': len(self.path_cache),
            'decision_history_length': len(self.decision_history)
        })
        return info


class SmartSnakeAI(SnakeAI):
    """超级智能贪吃蛇AI - 最高级版本"""
    
    def __init__(self, name: str = "SmartSnakeAI", player_id: int = 1):
        super().__init__(name, player_id)
        self.opponent_prediction_depth = 5
        self.territorial_control = True
        self.learning_enabled = True
        
        # 学习组件
        self.situation_memory = {}
        self.success_patterns = []
        self.failure_patterns = []
        
        # 高级策略参数
        self.dynamic_aggression = True
        self.adaptive_safety = True
        self.pattern_recognition = True
        
    def get_action(self, observation: Any, env: Any) -> Tuple[int, int]:
        """使用超级智能策略的动作选择"""
        start_time = time.time()
        
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return (0, 1)
        
        game = env.game
        if self.player_id == 1:
            snake = game.snake1
            opponent_snake = game.snake2
            alive = game.alive1
        else:
            snake = game.snake2
            opponent_snake = game.snake1
            alive = game.alive2
        
        if not snake or not alive:
            return random.choice(valid_actions)
        
        # 动态调整参数
        self._adapt_parameters(snake, opponent_snake, game)
        
        # 超级智能决策
        action = self._super_intelligent_decision(snake, opponent_snake, game, valid_actions)
        
        # 学习和记忆
        if self.learning_enabled:
            self._learn_from_situation(snake, opponent_snake, game, action)
        
        move_time = time.time() - start_time
        self.total_moves += 1
        self.total_time += move_time
        
        return action
    
    def _super_intelligent_decision(self, snake: List[Tuple[int, int]], 
                                  opponent_snake: List[Tuple[int, int]], 
                                  game: Any, valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """超级智能决策系统"""
        head = snake[0]
        
        # 1. 模式识别
        pattern_action = self._pattern_recognition_decision(snake, opponent_snake, game, valid_actions)
        if pattern_action:
            return pattern_action
        
        # 2. 深度预测策略
        prediction_action = self._deep_prediction_strategy(snake, opponent_snake, game, valid_actions)
        if prediction_action:
            return prediction_action
        
        # 3. 多步前瞻策略
        lookahead_action = self._multi_step_lookahead(snake, opponent_snake, game, valid_actions)
        if lookahead_action:
            return lookahead_action
        
        # 4. 回退到基础智能策略
        return self._strategic_decision(head, snake, opponent_snake, game, valid_actions, 
                                      game.direction1 if self.player_id == 1 else game.direction2)
    
    def _pattern_recognition_decision(self, snake: List[Tuple[int, int]], 
                                    opponent_snake: List[Tuple[int, int]], 
                                    game: Any, valid_actions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """基于模式识别的决策"""
        if not self.pattern_recognition:
            return None
        
        current_situation = self._encode_situation(snake, opponent_snake, game)
        
        # 查找相似的成功模式
        for pattern in self.success_patterns:
            if self._pattern_similarity(current_situation, pattern['situation']) > 0.8:
                # 找到相似的成功模式，尝试重复成功的动作
                if pattern['action'] in valid_actions:
                    return pattern['action']
        
        return None
    
    def _deep_prediction_strategy(self, snake: List[Tuple[int, int]], 
                                opponent_snake: List[Tuple[int, int]], 
                                game: Any, valid_actions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """深度预测策略"""
        if not opponent_snake:
            return None
        
        # 预测对手多步行为
        opponent_predictions = self._predict_opponent_behavior_deep(opponent_snake, game, self.opponent_prediction_depth)
        
        # 评估每个动作在预测场景下的表现
        action_scores = {}
        
        for action in valid_actions:
            score = 0.0
            
            for prediction_weight, future_opponent_positions in opponent_predictions:
                # 模拟执行这个动作后的情况
                simulation_score = self._simulate_action_outcome(snake, future_opponent_positions, game, action)
                score += simulation_score * prediction_weight
            
            action_scores[action] = score
        
        if action_scores:
            best_action = max(action_scores, key=action_scores.get)
            if action_scores[best_action] > 0:
                return best_action
        
        return None
    
    def _multi_step_lookahead(self, snake: List[Tuple[int, int]], 
                            opponent_snake: List[Tuple[int, int]], 
                            game: Any, valid_actions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """多步前瞻策略"""
        best_action = None
        best_score = float('-inf')
        
        for action in valid_actions:
            # 模拟执行这个动作3步
            score = self._evaluate_action_sequence(snake, opponent_snake, game, action, depth=3)
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action if best_score > float('-inf') else None
    
    def _adapt_parameters(self, snake: List[Tuple[int, int]], 
                        opponent_snake: List[Tuple[int, int]], game: Any):
        """动态调整策略参数"""
        if not self.dynamic_aggression:
            return
        
        # 根据当前情况调整攻击性
        if opponent_snake:
            length_advantage = len(snake) - len(opponent_snake)
            
            if length_advantage > 3:
                self.aggression_level = min(0.8, self.aggression_level + 0.1)
            elif length_advantage < -3:
                self.aggression_level = max(0.1, self.aggression_level - 0.1)
        
        # 根据剩余空间调整安全阈值
        if self.adaptive_safety:
            total_space = game.board_size * game.board_size
            occupied_space = len(snake) + (len(opponent_snake) if opponent_snake else 0)
            free_ratio = (total_space - occupied_space) / total_space
            
            if free_ratio < 0.3:  # 空间紧张
                self.safety_threshold = min(0.9, self.safety_threshold + 0.1)
            elif free_ratio > 0.7:  # 空间充足
                self.safety_threshold = max(0.4, self.safety_threshold - 0.1)
    
    def _predict_opponent_behavior_deep(self, opponent_snake: List[Tuple[int, int]], 
                                      game: Any, depth: int) -> List[Tuple[float, List[Tuple[int, int]]]]:
        """深度预测对手行为"""
        predictions = []
        
        if not opponent_snake or depth <= 0:
            return [(1.0, [])]
        
        opponent_head = opponent_snake[0]
        
        # 基于不同策略的预测
        strategies = [
            ('food_seeking', 0.4),
            ('survival', 0.3),
            ('aggressive', 0.2),
            ('random', 0.1)
        ]
        
        for strategy, weight in strategies:
            predicted_positions = self._simulate_opponent_strategy(opponent_head, strategy, game, depth)
            predictions.append((weight, predicted_positions))
        
        return predictions
    
    def _simulate_opponent_strategy(self, start_pos: Tuple[int, int], strategy: str, 
                                  game: Any, steps: int) -> List[Tuple[int, int]]:
        """模拟对手策略"""
        positions = [start_pos]
        current_pos = start_pos
        
        for _ in range(steps):
            if strategy == 'food_seeking':
                next_pos = self._predict_food_seeking_move(current_pos, game)
            elif strategy == 'survival':
                next_pos = self._predict_survival_move(current_pos, game)
            elif strategy == 'aggressive':
                next_pos = self._predict_aggressive_move(current_pos, game)
            else:  # random
                next_pos = self._predict_random_move(current_pos, game)
            
            if next_pos:
                positions.append(next_pos)
                current_pos = next_pos
            else:
                break
        
        return positions
    
    def _simulate_action_outcome(self, snake: List[Tuple[int, int]], 
                               future_opponent_positions: List[Tuple[int, int]], 
                               game: Any, action: Tuple[int, int]) -> float:
        """模拟动作结果"""
        head = snake[0]
        new_head = (head[0] + action[0], head[1] + action[1])
        
        score = 0.0
        
        # 基本安全检查
        if not self._is_valid_position_basic(new_head, game):
            return -1000.0
        
        # 与预测的对手位置冲突检查
        if new_head in future_opponent_positions:
            score -= 500.0
        
        # 空间控制评估
        controlled_space = self._quick_space_check(new_head, snake, [], game)
        score += controlled_space * 0.5
        
        # 食物接近度
        if game.foods:
            min_food_distance = min(self._manhattan_distance(new_head, food) for food in game.foods)
            score += 50 / (min_food_distance + 1)
        
        return score
    
    def _evaluate_action_sequence(self, snake: List[Tuple[int, int]], 
                                opponent_snake: List[Tuple[int, int]], 
                                game: Any, first_action: Tuple[int, int], depth: int) -> float:
        """评估动作序列"""
        if depth <= 0:
            return 0.0
        
        head = snake[0]
        new_head = (head[0] + first_action[0], head[1] + first_action[1])
        
        # 检查第一步是否安全
        if not self._is_position_safe_immediate(new_head, snake, opponent_snake, game):
            return float('-inf')
        
        # 递归评估后续步骤
        new_snake = [new_head] + snake[:-1]  # 简化的蛇移动
        
        # 获取下一步的可能动作
        next_actions = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (new_head[0] + dr, new_head[1] + dc)
            if self._is_position_safe_immediate(next_pos, new_snake, opponent_snake, game):
                next_actions.append((dr, dc))
        
        if not next_actions:
            return -100.0  # 死路
        
        # 评估最佳后续动作
        best_future_score = float('-inf')
        for next_action in next_actions:
            future_score = self._evaluate_action_sequence(new_snake, opponent_snake, game, next_action, depth - 1)
            best_future_score = max(best_future_score, future_score)
        
        # 当前步骤的即时奖励
        immediate_reward = self._calculate_immediate_reward(new_head, snake, game)
        
        return immediate_reward + 0.9 * best_future_score  # 折扣因子
    
    def _calculate_immediate_reward(self, position: Tuple[int, int], 
                                  snake: List[Tuple[int, int]], game: Any) -> float:
        """计算即时奖励"""
        reward = 0.0
        
        # 食物奖励
        if position in game.foods:
            reward += 100.0
        
        # 接近食物奖励
        if game.foods:
            min_distance = min(self._manhattan_distance(position, food) for food in game.foods)
            reward += 20.0 / (min_distance + 1)
        
        # 空间奖励
        space = self._quick_space_check(position, snake, [], game)
        reward += space * 0.1
        
        return reward
    
    def _encode_situation(self, snake: List[Tuple[int, int]], 
                        opponent_snake: List[Tuple[int, int]], game: Any) -> Dict[str, Any]:
        """编码当前情况"""
        head = snake[0]
        
        situation = {
            'my_length': len(snake),
            'opponent_length': len(opponent_snake) if opponent_snake else 0,
            'food_count': len(game.foods),
            'head_distance_to_center': self._manhattan_distance(head, (game.board_size // 2, game.board_size // 2)),
            'foods_nearby': sum(1 for food in game.foods if self._manhattan_distance(head, food) <= 3),
            'border_proximity': min(head[0], head[1], game.board_size - 1 - head[0], game.board_size - 1 - head[1])
        }
        
        if opponent_snake:
            opponent_head = opponent_snake[0]
            situation['opponent_distance'] = self._manhattan_distance(head, opponent_head)
        else:
            situation['opponent_distance'] = 999
        
        return situation
    
    def _pattern_similarity(self, situation1: Dict[str, Any], situation2: Dict[str, Any]) -> float:
        """计算模式相似度"""
        total_similarity = 0.0
        count = 0
        
        for key in situation1:
            if key in situation2:
                value1, value2 = situation1[key], situation2[key]
                if value1 == 0 and value2 == 0:
                    similarity = 1.0
                else:
                    similarity = 1.0 - abs(value1 - value2) / max(abs(value1), abs(value2), 1)
                total_similarity += similarity
                count += 1
        
        return total_similarity / count if count > 0 else 0.0
    
    def _learn_from_situation(self, snake: List[Tuple[int, int]], 
                            opponent_snake: List[Tuple[int, int]], 
                            game: Any, action: Tuple[int, int]):
        """从当前情况学习"""
        situation = self._encode_situation(snake, opponent_snake, game)
        
        # 简单的学习：记住当前情况和动作
        learning_entry = {
            'situation': situation,
            'action': action,
            'timestamp': time.time()
        }
        
        # 这里可以实现更复杂的学习逻辑
        # 比如根据后续的游戏结果来判断这个决策是否成功
        
    def _predict_food_seeking_move(self, pos: Tuple[int, int], game: Any) -> Optional[Tuple[int, int]]:
        """预测寻食移动"""
        if not game.foods:
            return None
        
        nearest_food = min(game.foods, key=lambda f: self._manhattan_distance(pos, f))
        
        dx = nearest_food[0] - pos[0]
        dy = nearest_food[1] - pos[1]
        
        if abs(dx) > abs(dy):
            if dx > 0:
                return (pos[0] + 1, pos[1])
            else:
                return (pos[0] - 1, pos[1])
        else:
            if dy > 0:
                return (pos[0], pos[1] + 1)
            else:
                return (pos[0], pos[1] - 1)
    
    def _predict_survival_move(self, pos: Tuple[int, int], game: Any) -> Optional[Tuple[int, int]]:
        """预测生存移动"""
        # 朝向空间最大的方向移动
        best_pos = None
        max_space = 0
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (pos[0] + dr, pos[1] + dc)
            if self._is_valid_position_basic(new_pos, game):
                space = self._quick_space_check(new_pos, [], [], game)
                if space > max_space:
                    max_space = space
                    best_pos = new_pos
        
        return best_pos
    
    def _predict_aggressive_move(self, pos: Tuple[int, int], game: Any) -> Optional[Tuple[int, int]]:
        """预测攻击性移动"""
        # 朝向中心移动
        center = (game.board_size // 2, game.board_size // 2)
        dx = center[0] - pos[0]
        dy = center[1] - pos[1]
        
        if abs(dx) > abs(dy):
            if dx > 0:
                return (pos[0] + 1, pos[1])
            else:
                return (pos[0] - 1, pos[1])
        else:
            if dy > 0:
                return (pos[0], pos[1] + 1)
            else:
                return (pos[0], pos[1] - 1)
    
    def _predict_random_move(self, pos: Tuple[int, int], game: Any) -> Optional[Tuple[int, int]]:
        """预测随机移动"""
        valid_moves = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (pos[0] + dr, pos[1] + dc)
            if self._is_valid_position_basic(new_pos, game):
                valid_moves.append(new_pos)
        
        return random.choice(valid_moves) if valid_moves else None
    
    def get_info(self) -> Dict[str, Any]:
        """获取SmartSnakeAI信息"""
        info = super().get_info()
        info.update({
            'type': 'Super Smart Snake AI',
            'description': '超级智能贪吃蛇AI，包含深度预测和学习能力',
            'strategy': 'Deep prediction + pattern recognition + multi-step lookahead',
            'opponent_prediction_depth': self.opponent_prediction_depth,
            'territorial_control': self.territorial_control,
            'learning_enabled': self.learning_enabled,
            'dynamic_aggression': self.dynamic_aggression,
            'adaptive_safety': self.adaptive_safety,
            'pattern_recognition': self.pattern_recognition,
            'success_patterns': len(self.success_patterns),
            'failure_patterns': len(self.failure_patterns),
            'situation_memory': len(self.situation_memory)
        })
        return info
