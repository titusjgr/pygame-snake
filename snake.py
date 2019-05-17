from pygame.locals import *
import pygame
import random

SCREEN_SIZE = 600, 480
MAP_SIZE = 60, 48 # This is the map for snake coordination.
SNAKE_SIZE = 10 # This indicates how big a block of the snake looks.
INITIAL_LENGTH = 4
BLACK = (  0  ,  0  ,  0  )
WHITE = ( 255 , 255 , 255 )
GREEN = (  0  , 255 ,  0  )

class Snake():
    def __init__(self, background):
        super().__init__()
       
        self.background = background

        self.direction = 'R' # To which direction the snake is moving
        self.invalid_direction = 'L'
        
        self.alive = True 
       
        # Every part of the snakes body
        self.positions = [
            (MAP_SIZE[0] // 2 - i, MAP_SIZE[1] // 2) for i in range(INITIAL_LENGTH)
        ]

        self.food_position = self.generate_food() 
        self.score = 0

        self.key_to_direction = {
            K_a: 'L',
            K_LEFT: 'L',
            K_d: 'R',
            K_RIGHT: 'R',
            K_w: 'U',
            K_UP: 'U',
            K_s: 'D',
            K_DOWN: 'D'
        }

        self.dir_to_invaliddir = {
            'L': 'R',
            'R': 'L',
            'U': 'D',
            'D': 'U',
        }

        self.move_direction = {
            'L': (-1,  0),
            'R': ( 1,  0),
            'U': ( 0, -1),
            'D': ( 0,  1)
        }

    def update(self):
        update_tuple = self.move_direction[self.direction]
        new_head = (self.positions[0][0] + update_tuple[0], self.positions[0][1] + update_tuple[1])
        
        self.positions.insert(0, new_head)

        if new_head == self.food_position:
            self.food_position = self.generate_food()
            self.score += 1
            reward = 1

        elif new_head in self.positions[1:-1] or not (0 <= new_head[0] < MAP_SIZE[0]) or not (0 <= new_head[1] < MAP_SIZE[1]):
            self.alive = False
            reward = -1

        else:
            self.positions.pop()

        self.invalid_direction = self.dir_to_invaliddir[self.direction]

        return reward
           

    def draw_body(self):
        for position in self.positions:
            pygame.draw.rect(self.background, WHITE, (
                    position[0] * SNAKE_SIZE,
                    position[1] * SNAKE_SIZE,
                    SNAKE_SIZE,
                    SNAKE_SIZE
                )
            )


    def draw_food(self):
        pygame.draw.rect(self.background, GREEN, (
            self.food_position[0] * SNAKE_SIZE,
            self.food_position[1] * SNAKE_SIZE,
            SNAKE_SIZE,
            SNAKE_SIZE
            )
        )

    def generate_food(self):
        while True: 
            food_pos = (
                random.randint(0, MAP_SIZE[0] - 1),
                random.randint(0, MAP_SIZE[1] - 1)
            )
            if food_pos not in self.positions:
                break
        return food_pos


