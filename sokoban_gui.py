"""
推箱子游戏专用GUI
"""

import pygame
import sys
import time
import os
from typing import Optional, Tuple, Dict, Any
from games.sokoban import SokobanGame, SokobanEnv, LEVELS
from agents import HumanAgent, RandomBot, SokobanAI, SmartSokobanAI, ExpertSokobanAI

# 颜色定义
COLORS = {
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'BROWN': (139, 69, 19),
    'LIGHT_BROWN': (205, 133, 63),
    'GRAY': (128, 128, 128),
    'LIGHT_GRAY': (211, 211, 211),
    'DARK_GRAY': (64, 64, 64),
    'RED': (255, 0, 0),
    'GREEN': (0, 255, 0),
    'BLUE': (0, 0, 255),
    'YELLOW': (255, 255, 0),
    'ORANGE': (255, 165, 0),
    'PURPLE': (128, 0, 128),
    'CYAN': (0, 255, 255),
    'DARK_GREEN': (0, 100, 0),
    'DARK_RED': (139, 0, 0)
}

class SokobanGUI:
    """推箱子图形界面 - 修复底部状态显示"""
    
    def __init__(self):
        # 初始化pygame
        pygame.init()
        
        # 布局参数优化
        self.cell_size = 28
        self.margin = 30
        self.ui_width = 320
        self.game_spacing = 40
        self.title_height = 35
        
        # 计算游戏区域大小
        max_width = max(len(level['layout'][0]) for level in LEVELS.values())
        max_height = max(len(level['layout']) for level in LEVELS.values())
        
        self.single_game_width = max_width * self.cell_size
        self.single_game_height = max_height * self.cell_size + self.title_height
        
        # 计算总窗口大小
        self.game_area_width = self.single_game_width * 2 + self.game_spacing + self.margin * 2
        self.game_area_height = self.single_game_height + self.margin * 2
        
        self.window_width = self.game_area_width + self.ui_width
        self.window_height = max(self.game_area_height, 750)  # 增加窗口高度
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Sokoban AI Battle")
        self.clock = pygame.time.Clock()
        
        # 字体
        self.font_large = pygame.font.Font(None, 30)
        self.font_medium = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 18)
        
        # 计算游戏区域位置
        self.player1_x = self.margin
        self.player2_x = self.margin + self.single_game_width + self.game_spacing
        self.game_y = self.margin
        
        # 游戏状态
        self.current_level = 'easy_1'
        self.env = SokobanEnv(level_name=self.current_level)
        self.human_agent = HumanAgent(name="Human Player", player_id=1)
        self.ai_agent = SokobanAI(name="Sokoban AI", player_id=2)
        self.current_agent = self.human_agent
        self.game_over = False
        self.winner = None
        self.thinking = False
        self.selected_ai = "SokobanAI"
        self.paused = False
        
        # UI元素
        self.buttons = self._create_buttons()
        
        # 游戏计时
        self.last_update = time.time()
        self.update_interval = 0.5
        
        self.reset_game()
    
    def _create_buttons(self) -> Dict[str, Dict[str, Any]]:
        """创建UI按钮 - 重新调整布局"""
        button_width = 140
        button_height = 26
        start_x = self.game_area_width + 20
        spacing = 4
        
        buttons = {
            # 关卡选择
            'easy_1': {
                'rect': pygame.Rect(start_x, 60, button_width, button_height),
                'text': 'Easy Level 1',
                'color': COLORS['YELLOW']
            },
            'easy_2': {
                'rect': pygame.Rect(start_x, 60 + (button_height + spacing) * 1, button_width, button_height),
                'text': 'Easy Level 2',
                'color': COLORS['LIGHT_GRAY']
            },
            'medium_1': {
                'rect': pygame.Rect(start_x, 60 + (button_height + spacing) * 2, button_width, button_height),
                'text': 'Medium Level 1',
                'color': COLORS['LIGHT_GRAY']
            },
            'hard_1': {
                'rect': pygame.Rect(start_x, 60 + (button_height + spacing) * 3, button_width, button_height),
                'text': 'Hard Level 1',
                'color': COLORS['LIGHT_GRAY']
            },
            
            # AI选择
            'basic_ai': {
                'rect': pygame.Rect(start_x, 200, button_width, button_height),
                'text': 'Basic AI',
                'color': COLORS['YELLOW']
            },
            'smart_ai': {
                'rect': pygame.Rect(start_x, 200 + (button_height + spacing) * 1, button_width, button_height),
                'text': 'Smart AI',
                'color': COLORS['LIGHT_GRAY']
            },
            'expert_ai': {
                'rect': pygame.Rect(start_x, 200 + (button_height + spacing) * 2, button_width, button_height),
                'text': 'Expert AI',
                'color': COLORS['LIGHT_GRAY']
            },
            
            # 控制按钮
            'new_game': {
                'rect': pygame.Rect(start_x, 310, button_width, button_height),
                'text': 'New Game',
                'color': COLORS['GREEN']
            },
            'pause': {
                'rect': pygame.Rect(start_x, 310 + (button_height + spacing) * 1, button_width, button_height),
                'text': 'Pause',
                'color': COLORS['ORANGE']
            },
            'quit': {
                'rect': pygame.Rect(start_x, 310 + (button_height + spacing) * 2, button_width, button_height),
                'text': 'Quit',
                'color': COLORS['RED']
            }
        }
        
        return buttons
    
    def _create_ai_agent(self):
        """创建AI智能体"""
        if self.selected_ai == "SokobanAI":
            self.ai_agent = SokobanAI(name="Basic AI", player_id=2)
        elif self.selected_ai == "SmartSokobanAI":
            self.ai_agent = SmartSokobanAI(name="Smart AI", player_id=2)
        elif self.selected_ai == "ExpertSokobanAI":
            self.ai_agent = ExpertSokobanAI(name="Expert AI", player_id=2)
    
    def reset_game(self):
        """重置游戏"""
        self.env = SokobanEnv(level_name=self.current_level)
        self.env.reset()
        self.game_over = False
        self.winner = None
        self.thinking = False
        self.current_agent = self.human_agent
        self.last_update = time.time()
        self.paused = False
        self.buttons['pause']['text'] = 'Pause'
    
    def handle_events(self) -> bool:
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                # 处理推箱子的键盘输入
                if (isinstance(self.current_agent, HumanAgent) and 
                    not self.game_over and not self.thinking and not self.paused):
                    self._handle_sokoban_input(event.key)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    mouse_pos = pygame.mouse.get_pos()
                    self._handle_button_click(mouse_pos)
        
        return True
    
    def _handle_button_click(self, mouse_pos: Tuple[int, int]):
        """处理按钮点击"""
        for button_name, button_info in self.buttons.items():
            if button_info['rect'].collidepoint(mouse_pos):
                if button_name == 'new_game':
                    self.reset_game()
                elif button_name == 'quit':
                    pygame.quit()
                    sys.exit()
                elif button_name == 'pause':
                    self.paused = not self.paused
                    self.buttons['pause']['text'] = 'Resume' if self.paused else 'Pause'
                elif button_name in LEVELS.keys():
                    # 切换关卡
                    for level_name in LEVELS.keys():
                        self.buttons[level_name]['color'] = COLORS['LIGHT_GRAY']
                    self.buttons[button_name]['color'] = COLORS['YELLOW']
                    self.current_level = button_name
                    self.reset_game()
                elif button_name.endswith('_ai'):
                    # 更新选中的AI
                    for ai_name in ['basic_ai', 'smart_ai', 'expert_ai']:
                        self.buttons[ai_name]['color'] = COLORS['LIGHT_GRAY']
                    
                    if button_name == 'basic_ai':
                        self.selected_ai = "SokobanAI"
                    elif button_name == 'smart_ai':
                        self.selected_ai = "SmartSokobanAI"
                    elif button_name == 'expert_ai':
                        self.selected_ai = "ExpertSokobanAI"
                    
                    self.buttons[button_name]['color'] = COLORS['YELLOW']
                    self._create_ai_agent()
                    self.reset_game()
    
    def _handle_sokoban_input(self, key):
        """处理推箱子键盘输入"""
        key_to_action = {
            pygame.K_UP: (-1, 0),    # 上
            pygame.K_w: (-1, 0),
            pygame.K_DOWN: (1, 0),   # 下
            pygame.K_s: (1, 0),
            pygame.K_LEFT: (0, -1),  # 左
            pygame.K_a: (0, -1),
            pygame.K_RIGHT: (0, 1),  # 右
            pygame.K_d: (0, 1)
        }
        
        if key in key_to_action:
            action = key_to_action[key]
            self._make_move(action)
    
    def _make_move(self, action):
        """执行移动"""
        if self.game_over or self.paused:
            return
        
        try:
            # 执行动作
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            # 检查游戏是否结束
            if terminated or truncated:
                self.game_over = True
                self.winner = self.env.get_winner()
            else:
                # 切换玩家
                self._switch_player()
        
        except Exception as e:
            print(f"Move execution failed: {e}")
    
    def _switch_player(self):
        """切换玩家"""
        if isinstance(self.current_agent, HumanAgent):
            self.current_agent = self.ai_agent
            self.thinking = True
        else:
            self.current_agent = self.human_agent
    
    def update_game(self):
        """更新游戏状态"""
        if self.game_over or self.paused:
            return
        
        current_time = time.time()
        
        # 检查是否需要更新
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        
        # AI回合
        if (not isinstance(self.current_agent, HumanAgent) and self.thinking):
            try:
                observation = self.env._get_observation()
                action = self.current_agent.get_action(observation, self.env)
                
                if action:
                    self._make_move(action)
                
                self.thinking = False
                
            except Exception as e:
                print(f"AI thinking failed: {e}")
                self.thinking = False
    
    def draw(self):
        """绘制游戏界面"""
        # 清空屏幕
        self.screen.fill(COLORS['WHITE'])
        
        # 绘制游戏区域
        self._draw_sokoban_game()
        
        # 绘制UI
        self._draw_ui()
        
        # 绘制游戏状态
        self._draw_game_status()
        
        # 更新显示
        pygame.display.flip()
    
    def _draw_sokoban_game(self):
        """绘制推箱子游戏"""
        observation = self.env._get_observation()
        
        # 绘制玩家1区域
        self._draw_game_grid(observation['grid1'], self.player1_x, self.game_y, "Player 1")
        
        # 绘制玩家2区域
        self._draw_game_grid(observation['grid2'], self.player2_x, self.game_y, "Player 2")
    
    def _draw_game_grid(self, grid, offset_x, offset_y, title):
        """绘制游戏网格"""
        # 绘制背景边框
        border_rect = pygame.Rect(
            offset_x - 5, 
            offset_y - 5, 
            self.single_game_width + 10, 
            self.single_game_height + 10
        )
        pygame.draw.rect(self.screen, COLORS['DARK_GRAY'], border_rect)
        pygame.draw.rect(self.screen, COLORS['BLACK'], border_rect, 2)
        
        # 绘制标题
        title_surface = self.font_medium.render(title, True, COLORS['BLACK'])
        title_rect = title_surface.get_rect(center=(offset_x + self.single_game_width // 2, offset_y + 15))
        self.screen.blit(title_surface, title_rect)
        
        # 颜色映射
        color_map = {
            0: COLORS['WHITE'],      # EMPTY
            1: COLORS['BLACK'],      # WALL
            2: COLORS['BLUE'],       # PLAYER
            3: COLORS['BROWN'],      # BOX
            4: COLORS['LIGHT_GRAY'], # TARGET
            5: COLORS['GREEN'],      # BOX_ON_TARGET
            6: COLORS['CYAN']        # PLAYER_ON_TARGET
        }
        
        # 绘制游戏网格
        grid_start_y = offset_y + self.title_height
        
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                x = offset_x + col * self.cell_size
                y = grid_start_y + row * self.cell_size
                
                cell_value = grid[row, col]
                color = color_map.get(cell_value, COLORS['GRAY'])
                
                # 绘制单元格
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLORS['BLACK'], rect, 1)
                
                # 绘制特殊标记
                center_x = x + self.cell_size // 2
                center_y = y + self.cell_size // 2
                
                if cell_value == 2:  # PLAYER
                    pygame.draw.circle(self.screen, COLORS['WHITE'], (center_x, center_y), self.cell_size // 4)
                elif cell_value == 3:  # BOX
                    pygame.draw.rect(self.screen, COLORS['DARK_GRAY'], 
                                   (x + 3, y + 3, self.cell_size - 6, self.cell_size - 6))
                elif cell_value == 4:  # TARGET
                    pygame.draw.circle(self.screen, COLORS['RED'], (center_x, center_y), self.cell_size // 6)
                elif cell_value == 5:  # BOX_ON_TARGET
                    pygame.draw.circle(self.screen, COLORS['RED'], (center_x, center_y), self.cell_size // 6)
                    pygame.draw.rect(self.screen, COLORS['DARK_GREEN'], 
                                   (x + 3, y + 3, self.cell_size - 6, self.cell_size - 6))
                elif cell_value == 6:  # PLAYER_ON_TARGET
                    pygame.draw.circle(self.screen, COLORS['RED'], (center_x, center_y), self.cell_size // 6)
                    pygame.draw.circle(self.screen, COLORS['WHITE'], (center_x, center_y), self.cell_size // 4)
    
    def _draw_ui(self):
        """绘制UI界面"""
        # 绘制UI背景
        ui_rect = pygame.Rect(self.game_area_width, 0, self.ui_width, self.window_height)
        pygame.draw.rect(self.screen, COLORS['LIGHT_GRAY'], ui_rect)
        pygame.draw.line(self.screen, COLORS['BLACK'], (self.game_area_width, 0), (self.game_area_width, self.window_height), 2)
        
        # 绘制按钮
        for button_name, button_info in self.buttons.items():
            pygame.draw.rect(self.screen, button_info['color'], button_info['rect'])
            pygame.draw.rect(self.screen, COLORS['BLACK'], button_info['rect'], 2)
            
            text_surface = self.font_small.render(button_info['text'], True, COLORS['BLACK'])
            text_rect = text_surface.get_rect(center=button_info['rect'].center)
            self.screen.blit(text_surface, text_rect)
        
        # 绘制标题
        start_x = self.game_area_width + 20
        
        level_title = self.font_medium.render("Level Selection:", True, COLORS['BLACK'])
        self.screen.blit(level_title, (start_x, 35))
        
        ai_title = self.font_medium.render("AI Selection:", True, COLORS['BLACK'])
        self.screen.blit(ai_title, (start_x, 175))
        
        # 绘制操作说明 - 调整位置
        instructions = [
            "Controls:",
            "• Arrow keys/WASD to move",
            "• Push boxes to targets",
            "• Complete level first to win",
            "",
            "Legend:",
            "• Blue circle: Player",
            "• Brown square: Box",
            "• Red dot: Target",
            "• Green square: Box on target"
        ]
        
        start_y = 400  # 调整起始位置
        for i, instruction in enumerate(instructions):
            text = self.font_small.render(instruction, True, COLORS['DARK_GRAY'])
            self.screen.blit(text, (start_x, start_y + i * 18))
    
    def _draw_game_status(self):
        """绘制游戏状态 - 重新排列避免重叠"""
        observation = self.env._get_observation()
        start_x = self.game_area_width + 20
        
        # 计算底部状态区域的起始位置
        status_start_y = self.window_height - 140  # 从底部往上140像素开始
        
        # 当前回合状态
        if self.paused:
            status_text = "Game Paused"
            color = COLORS['ORANGE']
        elif self.game_over:
            if self.winner == 1:
                status_text = "You Win!"
                color = COLORS['GREEN']
            elif self.winner == 2:
                status_text = "AI Wins!"
                color = COLORS['RED']
            else:
                status_text = "Draw!"
                color = COLORS['ORANGE']
        else:
            if isinstance(self.current_agent, HumanAgent):
                status_text = "Your Turn"
                color = COLORS['BLUE']
            else:
                if self.thinking:
                    status_text = "AI Thinking..."
                    color = COLORS['ORANGE']
                else:
                    status_text = "AI Turn"
                    color = COLORS['RED']
        
        # 绘制当前状态
        status_surface = self.font_medium.render(status_text, True, color)
        self.screen.blit(status_surface, (start_x, status_start_y))
        
        # 绘制分隔线
        pygame.draw.line(self.screen, COLORS['DARK_GRAY'], 
                        (start_x, status_start_y + 25), 
                        (start_x + 200, status_start_y + 25), 1)
        
        # 进度信息
        boxes1 = observation['boxes_in_place1']
        boxes2 = observation['boxes_in_place2']
        target = observation['target_boxes']
        moves1 = observation['moves1']
        moves2 = observation['moves2']
        
        # 玩家1进度
        progress1_text = f"Player 1: {boxes1}/{target} boxes"
        progress1_moves = f"Moves: {moves1}"
        
        progress1_surface = self.font_small.render(progress1_text, True, COLORS['BLUE'])
        progress1_moves_surface = self.font_small.render(progress1_moves, True, COLORS['BLUE'])
        
        self.screen.blit(progress1_surface, (start_x, status_start_y + 35))
        self.screen.blit(progress1_moves_surface, (start_x, status_start_y + 50))
        
        # 玩家2进度
        progress2_text = f"Player 2: {boxes2}/{target} boxes"
        progress2_moves = f"Moves: {moves2}"
        
        progress2_surface = self.font_small.render(progress2_text, True, COLORS['RED'])
        progress2_moves_surface = self.font_small.render(progress2_moves, True, COLORS['RED'])
        
        self.screen.blit(progress2_surface, (start_x, status_start_y + 70))
        self.screen.blit(progress2_moves_surface, (start_x, status_start_y + 85))
    
    def run(self):
        """运行游戏主循环"""
        running = True
        
        while running:
            # 处理事件
            running = self.handle_events()
            
            # 更新游戏
            self.update_game()
            
            # 绘制界面
            self.draw()
            
            # 控制帧率
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()


def main():
    """主函数"""
    print("启动推箱子AI对战平台...")
    print("操作说明:")
    print("- 方向键或WASD控制移动")
    print("- 推箱子到目标位置")
    print("- 最先完成关卡的玩家获胜")
    print("- 选择不同难度关卡和AI对手")
    
    try:
        game = SokobanGUI()
        game.run()
    except Exception as e:
        print(f"游戏错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
