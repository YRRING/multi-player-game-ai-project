"""
贪吃蛇游戏专用GUI
"""

import pygame
import sys
import time
from typing import Dict, Any
from games.snake import SnakeEnv
from agents import RandomBot
from agents.ai_bots.snake_ai import SnakeAI, BasicSnakeAI
from enum import Enum

# 初始化pygame
pygame.init()

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)

# 游戏常量
CELL_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
MARGIN = 50
UI_WIDTH = 200

# 方向枚举
class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

# 贪吃蛇游戏类
class SnakeGame:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.snake1 = [(5, 10), (5, 11), (5, 12)]  # 玩家1的蛇
        self.snake2 = [(15, 10), (15, 9), (15, 8)]  # 玩家2的蛇
        self.direction1 = Direction.RIGHT.value
        self.direction2 = Direction.LEFT.value
        self.food = self.generate_food()
        self.score1 = 0
        self.score2 = 0
        self.game_over = False
        self.winner = None
        self.move_count = 0
    
    def generate_food(self):
        while True:
            food = (random.randint(0, GRID_HEIGHT-1), random.randint(0, GRID_WIDTH-1))
            if food not in self.snake1 and food not in self.snake2:
                return food
    
    def move_snake(self, snake, direction):
        head = snake[0]
        new_head = (head[0] + direction[0], head[1] + direction[1])
        
        # 检查是否撞墙
        if (new_head[0] < 0 or new_head[0] >= GRID_HEIGHT or 
            new_head[1] < 0 or new_head[1] >= GRID_WIDTH):
            return False
        
        # 检查是否撞到自己或其他蛇
        if new_head in snake or new_head in self.snake1 or new_head in self.snake2:
            return False
        
        snake.insert(0, new_head)
        
        # 检查是否吃到食物
        if new_head == self.food:
            self.food = self.generate_food()
            return True
        else:
            snake.pop()
            return True
    
    def update(self, action1, action2):
        if self.game_over:
            return
        
        self.move_count += 1
        self.direction1 = action1 if action1 else self.direction1
        self.direction2 = action2 if action2 else self.direction2
        
        # 移动蛇
        snake1_alive = self.move_snake(self.snake1, self.direction1)
        snake2_alive = self.move_snake(self.snake2, self.direction2)
        
        # 检查游戏结束条件
        if not snake1_alive and not snake2_alive:
            self.game_over = True
            self.winner = -1  # 平局
        elif not snake1_alive:
            self.game_over = True
            self.winner = 2   # 玩家2赢
            self.score2 += 1
        elif not snake2_alive:
            self.game_over = True
            self.winner = 1   # 玩家1赢
            self.score1 += 1

# 简单AI
class SimpleAI:
    def get_action(self, game_state):
        snake = game_state["snake1"]
        food = game_state["food"]
        head = snake[0]
        
        # 简单的AI逻辑: 朝食物方向移动
        if food[0] < head[0] and Direction.UP.value not in [(-d[0], -d[1]) for d in game_state["possible_directions"]]:
            return Direction.UP.value
        elif food[0] > head[0] and Direction.DOWN.value not in [(-d[0], -d[1]) for d in game_state["possible_directions"]]:
            return Direction.DOWN.value
        elif food[1] < head[1] and Direction.LEFT.value not in [(-d[0], -d[1]) for d in game_state["possible_directions"]]:
            return Direction.LEFT.value
        elif food[1] > head[1] and Direction.RIGHT.value not in [(-d[0], -d[1]) for d in game_state["possible_directions"]]:
            return Direction.RIGHT.value
        
        # 如果没有明确方向，随机选择
        return random.choice(list(Direction)).value

# 游戏GUI类
class SnakeGUI:
    def __init__(self):
        self.screen = pygame.display.set_mode(
            (GRID_WIDTH * CELL_SIZE + MARGIN * 2 + UI_WIDTH, 
             GRID_HEIGHT * CELL_SIZE + MARGIN * 2)
        )
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.large_font = pygame.font.SysFont(None, 36)
        
        self.game = SnakeGame()
        self.ai = SimpleAI()
        self.human_direction = None
        self.paused = False
        self.last_update = 0
        self.update_interval = 0.15
        
        # 创建按钮
        self.buttons = {
            "new": {"rect": pygame.Rect(GRID_WIDTH * CELL_SIZE + MARGIN * 2 + 20, 50, 160, 40), "text": "New Game"},
            "pause": {"rect": pygame.Rect(GRID_WIDTH * CELL_SIZE + MARGIN * 2 + 20, 100, 160, 40), "text": "Pause"},
            "quit": {"rect": pygame.Rect(GRID_WIDTH * CELL_SIZE + MARGIN * 2 + 20, 150, 160, 40), "text": "Quit"}
        }
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    self.human_direction = Direction.UP.value
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    self.human_direction = Direction.DOWN.value
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    self.human_direction = Direction.LEFT.value
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    self.human_direction = Direction.RIGHT.value
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                    self.buttons["pause"]["text"] = "Resume" if self.paused else "Pause"
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                for name, btn in self.buttons.items():
                    if btn["rect"].collidepoint(event.pos):
                        if name == "new":
                            self.game.reset()
                        elif name == "pause":
                            self.paused = not self.paused
                            btn["text"] = "Resume" if self.paused else "Pause"
                        elif name == "quit":
                            return False
        
        return True
    
    def update(self):
        if self.paused or self.game.game_over:
            return
        
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.last_update = current_time
            
            # 获取AI动作
            game_state = {
                "snake1": self.game.snake1,
                "snake2": self.game.snake2,
                "food": self.game.food,
                "possible_directions": [d.value for d in Direction]
            }
            ai_action = self.ai.get_action(game_state)
            
            # 更新游戏状态
            self.game.update(self.human_direction, ai_action)
    
    def draw(self):
        self.screen.fill(GRAY)
        
        # 绘制游戏区域
        pygame.draw.rect(
            self.screen, BLACK, 
            (MARGIN, MARGIN, GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE)
        )
        
        # 绘制格子线
        for x in range(GRID_WIDTH + 1):
            pygame.draw.line(
                self.screen, (50, 50, 50), 
                (MARGIN + x * CELL_SIZE, MARGIN), 
                (MARGIN + x * CELL_SIZE, MARGIN + GRID_HEIGHT * CELL_SIZE)
            )
        for y in range(GRID_HEIGHT + 1):
            pygame.draw.line(
                self.screen, (50, 50, 50), 
                (MARGIN, MARGIN + y * CELL_SIZE), 
                (MARGIN + GRID_WIDTH * CELL_SIZE, MARGIN + y * CELL_SIZE)
            )
        
        # 绘制蛇
        for segment in self.game.snake1:
            pygame.draw.rect(
                self.screen, BLUE, 
                (MARGIN + segment[1] * CELL_SIZE + 2, MARGIN + segment[0] * CELL_SIZE + 2, 
                 CELL_SIZE - 4, CELL_SIZE - 4)
            )
        
        for segment in self.game.snake2:
            pygame.draw.rect(
                self.screen, RED, 
                (MARGIN + segment[1] * CELL_SIZE + 2, MARGIN + segment[0] * CELL_SIZE + 2, 
                 CELL_SIZE - 4, CELL_SIZE - 4)
            )
        
        # 绘制食物
        pygame.draw.rect(
            self.screen, GREEN, 
            (MARGIN + self.game.food[1] * CELL_SIZE + 2, MARGIN + self.game.food[0] * CELL_SIZE + 2, 
             CELL_SIZE - 4, CELL_SIZE - 4)
        )
        
        # 绘制UI面板
        ui_x = GRID_WIDTH * CELL_SIZE + MARGIN * 2
        pygame.draw.rect(
            self.screen, BLACK, 
            (ui_x, 0, UI_WIDTH, GRID_HEIGHT * CELL_SIZE + MARGIN * 2)
        )
        
        # 绘制按钮
        for name, btn in self.buttons.items():
            pygame.draw.rect(self.screen, WHITE, btn["rect"], border_radius=5)
            text = self.font.render(btn["text"], True, BLACK)
            self.screen.blit(
                text, 
                (btn["rect"].x + btn["rect"].width // 2 - text.get_width() // 2, 
                 btn["rect"].y + btn["rect"].height // 2 - text.get_height() // 2)
            )
        
        # 绘制分数
        score_text = self.font.render(f"Player: {self.game.score1}", True, BLUE)
        self.screen.blit(score_text, (ui_x + 20, 220))
        score_text = self.font.render(f"AI: {self.game.score2}", True, RED)
        self.screen.blit(score_text, (ui_x + 20, 250))
        moves_text = self.font.render(f"Moves: {self.game.move_count}", True, WHITE)
        self.screen.blit(moves_text, (ui_x + 20, 280))
        
        # 绘制游戏状态
        if self.game.game_over:
            if self.game.winner == 1:
                text = self.large_font.render("You Win!", True, BLUE)
            elif self.game.winner == 2:
                text = self.large_font.render("AI Wins!", True, RED)
            else:
                text = self.large_font.render("Draw!", True, YELLOW)
            self.screen.blit(
                text, 
                (MARGIN + GRID_WIDTH * CELL_SIZE // 2 - text.get_width() // 2, 10)
            )
        elif self.paused:
            text = self.large_font.render("Paused", True, YELLOW)
            self.screen.blit(
                text, 
                (MARGIN + GRID_WIDTH * CELL_SIZE // 2 - text.get_width() // 2, 10)
            )
        
        pygame.display.flip()
    
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

# 启动游戏
if __name__ == "__main__":
    game = SnakeGUI()
    game.run()

if __name__ == "__main__":
    main() 
