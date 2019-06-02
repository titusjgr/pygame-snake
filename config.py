MAP_SIZE = 10, 10  # This is the map for snake coordination.
SNAKE_SIZE = 15  # This indicates how big a block of the snake looks to the net.
SCREEN_SIZE = MAP_SIZE[0] * SNAKE_SIZE, MAP_SIZE[1] * SNAKE_SIZE
INITIAL_LENGTH = 4
FPS = 5
NUM_FRAMES = 4  # num of frames used as input state at a time
DISCOUNT_FACTOR = 0.97
INPUT_SHAPE = SCREEN_SIZE[0], SCREEN_SIZE[1], NUM_FRAMES
SNAKE_DISPLAY_SIZE = 30
DISPLAY_SIZE = (
    MAP_SIZE[0] * SNAKE_DISPLAY_SIZE,
    MAP_SIZE[1] * SNAKE_DISPLAY_SIZE
)
