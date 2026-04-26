# window size
WIDTH = 800
HEIGHT = 400
PANEL_HEIGHT = 120
FPS = 60

# paddle and ball dimensions
PADDLE_WIDTH = 12
PADDLE_HEIGHT = 80
PADDLE_SPEED = 5

BALL_SIZE = 10
BALL_SPEED = 5

# how many bins (q table)
N_BALL_X = 20
N_BALL_Y = 10
N_BALL_VX = 2
N_BALL_VY = 5
N_PADDLE_Y = 10

# q-learning settings
ALPHA = 0.15
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9998 

# rewards
REWARD_HIT = 1.0
REWARD_MISS = -5.0

# save a checkpoint at each of these rally counts
RALLY_CHECKPOINTS = [500,15000,30000]

# file paths
MODEL_DIR = "models"
MODEL_LEFT = "models/left_agent.pkl"
MODEL_RIGHT = "models/right_agent.pkl"

CKPT_LEFT = "models/left_agent_rally_{}.pkl"
CKPT_RIGHT = "models/right_agent_rally_{}.pkl"