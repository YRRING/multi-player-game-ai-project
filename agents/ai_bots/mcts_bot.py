"""
MCTS Bot - 完善优化版本
使用蒙特卡洛树搜索算法，增强状态管理和模拟策略
"""

import time
import random
import math
from typing import Dict, List, Tuple, Any, Optional
from agents.base_agent import BaseAgent
import config
import copy
import numpy as np


class MCTSNode:
    """增强的MCTS节点"""
    
    def __init__(self, state, parent=None, action=None, player_id=1):
        self.state = state
        self.parent = parent
        self.action = action
        self.player_id = player_id
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None
        self.is_expanded = False
        self.is_terminal_node = None
        self.winner = None
        
        # 延迟初始化未尝试的动作
        self._init_untried_actions()
    
    def _init_untried_actions(self):
        """初始化未尝试的动作"""
        try:
            if hasattr(self.state, 'get_valid_actions'):
                self.untried_actions = list(self.state.get_valid_actions())
            elif hasattr(self.state, 'game') and hasattr(self.state.game, 'get_valid_actions'):
                self.untried_actions = list(self.state.game.get_valid_actions())
            else:
                self.untried_actions = []
        except Exception as e:
            self.untried_actions = []
    
    def is_fully_expanded(self):
        """检查是否完全展开"""
        return len(self.untried_actions) == 0 and self.is_expanded
    
    def is_terminal(self):
        """检查是否为终止节点（缓存结果）"""
        if self.is_terminal_node is None:
            try:
                if hasattr(self.state, 'is_terminal'):
                    self.is_terminal_node = self.state.is_terminal()
                elif hasattr(self.state, 'game') and hasattr(self.state.game, 'is_terminal'):
                    self.is_terminal_node = self.state.game.is_terminal()
                else:
                    self.is_terminal_node = False
            except:
                self.is_terminal_node = False
        
        return self.is_terminal_node
    
    def get_winner(self):
        """获取获胜者（缓存结果）"""
        if self.winner is None:
            try:
                if hasattr(self.state, 'get_winner'):
                    self.winner = self.state.get_winner()
                elif hasattr(self.state, 'game') and hasattr(self.state.game, 'get_winner'):
                    self.winner = self.state.game.get_winner()
                else:
                    self.winner = None
            except:
                self.winner = None
        
        return self.winner
    
    def ucb1_value(self, exploration_constant=1.414, parent_visits=None):
        """计算增强的UCB1值"""
        if self.visits == 0:
            return float('inf')
        
        if parent_visits is None:
            parent_visits = self.parent.visits if self.parent else 1
        
        exploitation = self.value / self.visits
        
        # 防止log(0)
        if parent_visits <= 0:
            exploration = 0
        else:
            exploration = exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)
        
        # 添加随机噪声打破平局
        noise = random.uniform(-0.001, 0.001)
        
        return exploitation + exploration + noise
    
    def add_child(self, child_state, action):
        """添加子节点"""
        child = MCTSNode(child_state, self, action, self.player_id)
        self.children.append(child)
        return child
    
    def update(self, reward):
        """更新节点统计"""
        self.visits += 1
        self.value += reward
    
    def robust_child(self):
        """选择访问次数最多的子节点"""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)
    
    def best_child(self, exploration_constant=1.414):
        """选择最佳子节点"""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb1_value(exploration_constant, self.visits))
    
    def get_state_hash(self):
        """获取状态哈希用于比较"""
        try:
            if hasattr(self.state, 'board'):
                return hash(self.state.board.tobytes())
            elif hasattr(self.state, 'snake1'):
                return hash((tuple(self.state.snake1), tuple(self.state.snake2), tuple(self.state.foods)))
            else:
                return hash(str(self.state))
        except:
            return random.randint(0, 1000000)


class MCTSBot(BaseAgent):
    """增强的MCTS Bot - 完善的蒙特卡洛树搜索实现"""
    
    def __init__(self, name: str = "MCTSBot", player_id: int = 1, 
                 simulation_count: int = 1000, exploration_constant: float = 1.414):
        super().__init__(name, player_id)
        self.simulation_count = simulation_count
        self.exploration_constant = exploration_constant
        self.tree_reuse = True
        self.root = None
        
        # 从配置获取参数
        ai_config = config.AI_CONFIGS.get('mcts', {})
        self.simulation_count = ai_config.get('simulation_count', simulation_count)
        self.exploration_constant = ai_config.get('exploration_constant', exploration_constant)
        self.timeout = ai_config.get('timeout', 10)
        
        # 增强参数
        self.use_progressive_widening = True
        self.use_rave = False  # RAVE (Rapid Action Value Estimation)
        self.use_early_termination = True
        self.adaptive_exploration = True
        
        # 统计信息
        self.total_simulations = 0
        self.tree_size = 0
        self.max_depth_reached = 0
        self.simulation_depths = []
        
        # RAVE相关
        self.rave_constant = 3000
        self.rave_stats = {}
    
    def get_action(self, observation: Any, env: Any) -> Any:
        """
        使用增强的MCTS选择动作
        """
        start_time = time.time()
        
        # 获取有效动作
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            return None
        
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        # 创建或更新根节点
        current_state = self._safe_clone_state(env)
        if current_state is None:
            return random.choice(valid_actions)
        
        # 树重用逻辑
        if self.root is None or not self.tree_reuse:
            self.root = MCTSNode(current_state, player_id=self.player_id)
        else:
            # 尝试在现有树中找到匹配的节点
            self.root = self._find_or_create_root(current_state)
        
        # 自适应探索常数
        if self.adaptive_exploration:
            self.exploration_constant = self._calculate_adaptive_exploration(valid_actions)
        
        # 执行MCTS搜索
        simulations_done = 0
        time_per_simulation = []
        
        while (simulations_done < self.simulation_count and 
               time.time() - start_time < self.timeout):
            
            sim_start = time.time()
            
            # MCTS的四个阶段
            leaf = self._select(self.root)
            
            if not leaf.is_terminal():
                leaf = self._expand(leaf)
            
            reward, simulation_depth = self._simulate(leaf)
            
            self._backpropagate(leaf, reward)
            
            # 记录统计
            sim_time = time.time() - sim_start
            time_per_simulation.append(sim_time)
            self.simulation_depths.append(simulation_depth)
            
            simulations_done += 1
            
            # 早期终止条件
            if (self.use_early_termination and 
                simulations_done > 100 and 
                self._should_terminate_early()):
                break
        
        # 选择最佳动作
        best_child = self._select_final_action()
        
        if best_child is None:
            return random.choice(valid_actions)
        
        # 更新统计
        move_time = time.time() - start_time
        self.total_moves += 1
        self.total_time += move_time
        self.total_simulations += simulations_done
        self.tree_size = self._count_tree_size(self.root)
        
        # 树重用：更新根节点
        if self.tree_reuse and best_child:
            self.root = best_child
            self.root.parent = None
        
        return best_child.action
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        增强的选择阶段
        """
        current = node
        depth = 0
        
        while not current.is_terminal() and current.is_fully_expanded():
            if not current.children:
                break
            
            # 使用UCB1或增强的选择策略
            current = current.best_child(self.exploration_constant)
            if current is None:
                break
            
            depth += 1
            
            # 防止无限深度
            if depth > 50:
                break
        
        self.max_depth_reached = max(self.max_depth_reached, depth)
        return current
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        增强的扩展阶段
        """
        if node.is_terminal():
            return node
        
        # 渐进式拓宽
        if self.use_progressive_widening:
            max_children = int(math.ceil(math.sqrt(node.visits + 1)))
            if len(node.children) >= max_children and node.untried_actions:
                # 还不到扩展新节点的时候
                return node
        
        # 如果还有未尝试的动作
        if node.untried_actions:
            action = self._select_expansion_action(node)
            if action in node.untried_actions:
                node.untried_actions.remove(action)
            
            # 创建新状态
            try:
                new_state = self._apply_action_to_state(node.state, action)
                if new_state is not None:
                    child = node.add_child(new_state, action)
                    return child
            except Exception as e:
                pass
        
        # 如果所有动作都已尝试，标记为已完全扩展
        if not node.untried_actions:
            node.is_expanded = True
        
        return node
    
    def _simulate(self, node: MCTSNode) -> Tuple[float, int]:
        """
        增强的模拟阶段
        """
        # 如果节点已经是终止状态
        if node.is_terminal():
            reward = self._evaluate_terminal_node(node)
            return reward, 0
        
        # 克隆状态进行模拟
        simulation_state = self._safe_clone_state(node.state)
        if simulation_state is None:
            return 0.0, 0
        
        # 记录模拟过程中的动作（用于RAVE）
        simulation_actions = []
        simulation_depth = 0
        max_simulation_depth = 100
        
        # 混合模拟策略
        heavy_playouts = node.visits < 10  # 对新节点使用更智能的模拟
        
        while (not self._is_state_terminal(simulation_state) and 
               simulation_depth < max_simulation_depth):
            
            valid_actions = self._get_state_valid_actions(simulation_state)
            if not valid_actions:
                break
            
            # 选择模拟动作
            if heavy_playouts:
                action = self._intelligent_simulation_policy(simulation_state, valid_actions)
            else:
                action = self._fast_simulation_policy(simulation_state, valid_actions)
            
            simulation_actions.append(action)
            
            # 执行动作
            try:
                simulation_state = self._apply_action_to_state(simulation_state, action)
                if simulation_state is None:
                    break
            except:
                break
            
            simulation_depth += 1
        
        # 评估最终状态
        reward = self._evaluate_simulation_result(simulation_state)
        
        # 更新RAVE统计
        if self.use_rave:
            self._update_rave_stats(simulation_actions, reward)
        
        return reward, simulation_depth
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        增强的回传阶段
        """
        current = node
        visits_from_root = 0
        
        while current is not None:
            # 根据当前玩家和节点深度调整奖励
            player_multiplier = self._get_player_multiplier(current)
            adjusted_reward = reward * player_multiplier
            
            # 应用折扣因子（距离根节点越远，影响越小）
            discount_factor = 0.95 ** visits_from_root
            final_reward = adjusted_reward * discount_factor
            
            current.update(final_reward)
            
            current = current.parent
            visits_from_root += 1
    
    def _intelligent_simulation_policy(self, state, valid_actions: List[Any]) -> Any:
        """智能模拟策略"""
        try:
            if hasattr(state, 'board') or (hasattr(state, 'game') and hasattr(state.game, 'board')):
                return self._gomoku_intelligent_policy(state, valid_actions)
            elif hasattr(state, 'snake1') or (hasattr(state, 'game') and hasattr(state.game, 'snake1')):
                return self._snake_intelligent_policy(state, valid_actions)
            else:
                return self._default_intelligent_policy(state, valid_actions)
        except:
            return random.choice(valid_actions)
    
    def _fast_simulation_policy(self, state, valid_actions: List[Any]) -> Any:
        """快速模拟策略"""
        # 80%随机，20%启发式
        if random.random() < 0.8:
            return random.choice(valid_actions)
        else:
            return self._intelligent_simulation_policy(state, valid_actions)
    
    def _gomoku_intelligent_policy(self, state, valid_actions: List[Any]) -> Any:
        """五子棋智能模拟策略"""
        try:
            board = state.board if hasattr(state, 'board') else state.game.board
            current_player = state.current_player if hasattr(state, 'current_player') else state.game.current_player
            
            # 1. 检查是否能获胜
            for action in valid_actions:
                if self._can_win_gomoku(board, action, current_player):
                    return action
            
            # 2. 检查是否需要阻止对手获胜
            opponent = 3 - current_player
            for action in valid_actions:
                if self._can_win_gomoku(board, action, opponent):
                    return action
            
            # 3. 寻找能形成威胁的位置
            threat_actions = []
            for action in valid_actions:
                threat_level = self._calculate_threat_level_gomoku(board, action, current_player)
                if threat_level > 0:
                    threat_actions.append((action, threat_level))
            
            if threat_actions:
                threat_actions.sort(key=lambda x: x[1], reverse=True)
                return threat_actions[0][0]
            
            # 4. 选择中心附近的位置
            center = board.shape[0] // 2
            center_actions = []
            for action in valid_actions:
                if isinstance(action, tuple) and len(action) == 2:
                    distance = abs(action[0] - center) + abs(action[1] - center)
                    if distance <= 4:
                        center_actions.append(action)
            
            if center_actions:
                return random.choice(center_actions)
        except:
            pass
        
        return random.choice(valid_actions)
    
    def _snake_intelligent_policy(self, state, valid_actions: List[Any]) -> Any:
        """贪吃蛇智能模拟策略"""
        try:
            game = state if hasattr(state, 'snake1') else state.game
            current_player = state.current_player if hasattr(state, 'current_player') else game.current_player
            
            my_snake = game.snake1 if current_player == 1 else game.snake2
            opponent_snake = game.snake2 if current_player == 1 else game.snake1
            
            if not my_snake:
                return random.choice(valid_actions)
            
            head = my_snake[0]
            safe_actions = []
            
            # 1. 过滤安全动作
            for action in valid_actions:
                new_head = (head[0] + action[0], head[1] + action[1])
                
                if self._is_safe_snake_position(new_head, game):
                    safe_actions.append(action)
            
            if not safe_actions:
                return random.choice(valid_actions)
            
            # 2. 优先朝食物移动
            if game.foods:
                food_actions = []
                nearest_food = min(game.foods, 
                                 key=lambda f: abs(head[0] - f[0]) + abs(head[1] - f[1]))
                
                for action in safe_actions:
                    new_head = (head[0] + action[0], head[1] + action[1])
                    old_distance = abs(head[0] - nearest_food[0]) + abs(head[1] - nearest_food[1])
                    new_distance = abs(new_head[0] - nearest_food[0]) + abs(new_head[1] - nearest_food[1])
                    
                    if new_distance < old_distance:
                        food_actions.append(action)
                
                if food_actions:
                    return random.choice(food_actions)
            
            # 3. 选择空间最大的方向
            space_actions = []
            max_space = 0
            
            for action in safe_actions:
                new_head = (head[0] + action[0], head[1] + action[1])
                space = self._calculate_space_around(new_head, game)
                
                if space > max_space:
                    max_space = space
                    space_actions = [action]
                elif space == max_space:
                    space_actions.append(action)
            
            if space_actions:
                return random.choice(space_actions)
            
            return random.choice(safe_actions)
        except:
            pass
        
        return random.choice(valid_actions)
    
    def _default_intelligent_policy(self, state, valid_actions: List[Any]) -> Any:
        """默认智能策略"""
        # 简单的启发式：避免明显的坏动作
        if len(valid_actions) <= 2:
            return random.choice(valid_actions)
        
        # 随机选择，但避免第一个和最后一个（通常是边界情况）
        middle_actions = valid_actions[1:-1] if len(valid_actions) > 2 else valid_actions
        return random.choice(middle_actions if middle_actions else valid_actions)
    
    def _can_win_gomoku(self, board: np.ndarray, action: Tuple[int, int], player: int) -> bool:
        """检查五子棋是否能获胜"""
        row, col = action
        if board[row, col] != 0:
            return False
        
        # 临时放置棋子
        board[row, col] = player
        
        # 检查四个方向
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1  # 当前位置
            
            # 正方向
            r, c = row + dr, col + dc
            while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
                   board[r, c] == player):
                count += 1
                r += dr
                c += dc
            
            # 负方向
            r, c = row - dr, col - dc
            while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
                   board[r, c] == player):
                count += 1
                r -= dr
                c -= dc
            
            if count >= 5:
                board[row, col] = 0  # 恢复
                return True
        
        board[row, col] = 0  # 恢复
        return False
    
    def _calculate_threat_level_gomoku(self, board: np.ndarray, action: Tuple[int, int], player: int) -> int:
        """计算五子棋威胁等级"""
        row, col = action
        if board[row, col] != 0:
            return 0
        
        board[row, col] = player
        threat_level = 0
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            
            # 正方向
            r, c = row + dr, col + dc
            while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
                   board[r, c] == player):
                count += 1
                r += dr
                c += dc
            
            # 负方向
            r, c = row - dr, col - dc
            while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
                   board[r, c] == player):
                count += 1
                r -= dr
                c -= dc
            
            if count >= 4:
                threat_level += 100
            elif count == 3:
                threat_level += 10
            elif count == 2:
                threat_level += 1
        
        board[row, col] = 0
        return threat_level
    
    def _is_safe_snake_position(self, position: Tuple[int, int], game: Any) -> bool:
        """检查贪吃蛇位置是否安全"""
        row, col = position
        
        # 边界检查
        if (row < 0 or row >= game.board_size or 
            col < 0 or col >= game.board_size):
            return False
        
        # 蛇身检查（排除尾巴，因为下一步会移动）
        if position in game.snake1[:-1] or position in game.snake2[:-1]:
            return False
        
        return True
    
    def _calculate_space_around(self, position: Tuple[int, int], game: Any) -> int:
        """计算周围空间"""
        visited = set()
        queue = [position]
        visited.add(position)
        
        while queue and len(visited) < 20:  # 限制搜索范围
            current = queue.pop(0)
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (current[0] + dr, current[1] + dc)
                
                if (new_pos not in visited and 
                    self._is_safe_snake_position(new_pos, game)):
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        return len(visited)
    
    def _select_expansion_action(self, node: MCTSNode) -> Any:
        """选择扩展动作"""
        if not node.untried_actions:
            return None
        
        # 可以添加更智能的选择策略
        return random.choice(node.untried_actions)
    
    def _should_terminate_early(self) -> bool:
        """早期终止判断"""
        if not self.root or not self.root.children:
            return False
        
        # 如果最佳选择已经明显
        visit_counts = [child.visits for child in self.root.children]
        max_visits = max(visit_counts)
        second_max = sorted(visit_counts, reverse=True)[1] if len(visit_counts) > 1 else 0
        
        # 如果最佳选择的访问次数远超第二名
        return max_visits > second_max * 3 and max_visits > 50
    
    def _select_final_action(self) -> Optional[MCTSNode]:
        """选择最终动作"""
        if not self.root or not self.root.children:
            return None
        
        # 使用robust child（访问次数最多）
        return self.root.robust_child()
    
    def _calculate_adaptive_exploration(self, valid_actions: List[Any]) -> float:
        """计算自适应探索常数"""
        base_exploration = self.exploration_constant
        
        # 根据动作数量调整
        action_count = len(valid_actions)
        if action_count > 50:
            return base_exploration * 1.5
        elif action_count < 10:
            return base_exploration * 0.7
        
        return base_exploration
    
    def _find_or_create_root(self, current_state) -> MCTSNode:
        """在现有树中查找或创建根节点"""
        # 简化实现：总是创建新根节点
        # 在实际应用中，可以通过状态哈希来匹配现有节点
        return MCTSNode(current_state, player_id=self.player_id)
    
    def _safe_clone_state(self, state) -> Any:
        """安全的状态克隆"""
        try:
            if hasattr(state, 'clone'):
                return state.clone()
            elif hasattr(state, 'game') and hasattr(state.game, 'clone'):
                cloned_env = type(state)()
                cloned_env.game = state.game.clone()
                return cloned_env
            else:
                return copy.deepcopy(state)
        except Exception as e:
            return None
    
    def _apply_action_to_state(self, state, action):
        """将动作应用到状态"""
        try:
            if hasattr(state, 'clone'):
                new_state = state.clone()
            else:
                new_state = copy.deepcopy(state)
            
            if hasattr(new_state, 'step'):
                new_state.step(action)
            elif hasattr(new_state, 'game') and hasattr(new_state.game, 'step'):
                new_state.game.step(action)
            
            return new_state
        except Exception as e:
            return None
    
    def _is_state_terminal(self, state) -> bool:
        """检查状态是否终止"""
        try:
            if hasattr(state, 'is_terminal'):
                return state.is_terminal()
            elif hasattr(state, 'game') and hasattr(state.game, 'is_terminal'):
                return state.game.is_terminal()
            else:
                return False
        except:
            return False
    
    def _get_state_valid_actions(self, state) -> List[Any]:
        """获取状态的有效动作"""
        try:
            if hasattr(state, 'get_valid_actions'):
                return state.get_valid_actions()
            elif hasattr(state, 'game') and hasattr(state.game, 'get_valid_actions'):
                return state.game.get_valid_actions()
            else:
                return []
        except:
            return []
    
    def _get_player_multiplier(self, node: MCTSNode) -> float:
        """获取玩家乘数"""
        try:
            if hasattr(node.state, 'current_player'):
                current_player = node.state.current_player
            elif hasattr(node.state, 'game') and hasattr(node.state.game, 'current_player'):
                current_player = node.state.game.current_player
            else:
                return 1.0
            
            return 1.0 if current_player == self.player_id else -1.0
        except:
            return 1.0
    
    def _evaluate_terminal_node(self, node: MCTSNode) -> float:
        """评估终止节点"""
        winner = node.get_winner()
        if winner == self.player_id:
            return 1.0
        elif winner is not None:
            return -1.0
        else:
            return 0.0
    
    def _evaluate_simulation_result(self, state) -> float:
        """评估模拟结果"""
        try:
            if self._is_state_terminal(state):
                if hasattr(state, 'get_winner'):
                    winner = state.get_winner()
                elif hasattr(state, 'game') and hasattr(state.game, 'get_winner'):
                    winner = state.game.get_winner()
                else:
                    winner = None
                
                if winner == self.player_id:
                    return 1.0
                elif winner is not None:
                    return -1.0
                else:
                    return 0.0
            else:
                # 非终止状态的启发式评估
                return self._heuristic_evaluation(state)
        except:
            return 0.0
    
    def _heuristic_evaluation(self, state) -> float:
        """启发式评估"""
        try:
            if hasattr(state, 'snake1') or (hasattr(state, 'game') and hasattr(state.game, 'snake1')):
                # 贪吃蛇游戏的启发式评估
                game = state if hasattr(state, 'snake1') else state.game
                
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
                
                if not my_alive:
                    return -1.0
                if not opponent_alive:
                    return 1.0
                
                if my_snake and opponent_snake:
                    length_diff = len(my_snake) - len(opponent_snake)
                    return max(-1.0, min(1.0, length_diff * 0.1))
            
            return random.uniform(-0.1, 0.1)
        except:
            return 0.0
    
    def _update_rave_stats(self, actions: List[Any], reward: float):
        """更新RAVE统计"""
        for action in actions:
            if action not in self.rave_stats:
                self.rave_stats[action] = {'wins': 0, 'visits': 0}
            
            self.rave_stats[action]['visits'] += 1
            if reward > 0:
                self.rave_stats[action]['wins'] += reward
    
    def _count_tree_size(self, node: MCTSNode) -> int:
        """计算树的大小"""
        if node is None:
            return 0
        
        count = 1
        for child in node.children:
            count += self._count_tree_size(child)
        
        return count
    
    def reset(self):
        """重置MCTS Bot"""
        super().reset()
        self.root = None
        self.total_simulations = 0
        self.tree_size = 0
        self.max_depth_reached = 0
        self.simulation_depths = []
        self.rave_stats.clear()
    
    def get_info(self) -> Dict[str, Any]:
        """获取MCTS Bot信息"""
        info = super().get_info()
        
        avg_sim_depth = np.mean(self.simulation_depths) if self.simulation_depths else 0
        
        info.update({
            'type': 'Enhanced MCTS',
            'description': '增强的蒙特卡洛树搜索算法',
            'strategy': f'Enhanced MCTS with {self.simulation_count} simulations',
            'simulation_count': self.simulation_count,
            'exploration_constant': self.exploration_constant,
            'timeout': self.timeout,
            'total_simulations': self.total_simulations,
            'tree_size': self.tree_size,
            'max_depth_reached': self.max_depth_reached,
            'avg_simulation_depth': avg_sim_depth,
            'tree_reuse': self.tree_reuse,
            'use_progressive_widening': self.use_progressive_widening,
            'use_rave': self.use_rave,
            'adaptive_exploration': self.adaptive_exploration,
            'rave_actions': len(self.rave_stats)
        })
        return info
