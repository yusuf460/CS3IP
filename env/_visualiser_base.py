import sys
import os
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import QLearningAgent
from game_env import PongEnv
from config import (
    WIDTH, HEIGHT, PANEL_HEIGHT, FPS,
    PADDLE_WIDTH, PADDLE_HEIGHT, BALL_SIZE,
    ALPHA, GAMMA, EPSILON_MIN, EPSILON_DECAY,
    REWARD_HIT, REWARD_MISS,
    RALLY_CHECKPOINTS,
)

# colours
BLACK    = (0, 0, 0)
WHITE    = (255, 255, 255)
GRAY     = (80, 80, 80)
DIM_GRAY = (45, 45, 55)
PANEL_BG = (12, 12, 20)
SEP_LINE = (50, 50, 70)
BLUE     = (50, 150, 255)
RED      = (255, 70, 70)
GREEN    = (80, 220, 100)
YELLOW   = (255, 220, 50)
TEAL     = (60, 200, 180)
LABEL_COL = (140, 140, 160)

MAX_TRAIL = 20


def draw_trail(screen, trail):
    for (tx, ty, age) in trail:
        alpha = int(age / MAX_TRAIL * 110)
        r = max(1, BALL_SIZE * age // MAX_TRAIL)
        col = (255, 255, max(80, 255 - (MAX_TRAIL - age) * 11))
        surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*col, alpha), (r, r), r)
        screen.blit(surf, (tx + BALL_SIZE // 2 - r, ty + BALL_SIZE // 2 - r))


def draw_game(screen, rs, trail, fonts, infinite_mode=False):
    screen.fill(BLACK)

    # dashed centre line
    for y in range(0, HEIGHT, 20):
        pygame.draw.rect(screen, GRAY, (WIDTH // 2 - 2, y, 4, 10))

    draw_trail(screen, trail)

    # ball
    ball_rect = pygame.Rect(int(rs["ball_x"]), int(rs["ball_y"]), BALL_SIZE, BALL_SIZE)
    pygame.draw.rect(screen, WHITE, ball_rect, border_radius=3)

    # paddles
    lp = pygame.Rect(30, int(rs["left_y"]), PADDLE_WIDTH, PADDLE_HEIGHT)
    rp = pygame.Rect(WIDTH - 30 - PADDLE_WIDTH, int(rs["right_y"]), PADDLE_WIDTH, PADDLE_HEIGHT)

    for rect, col in [(lp, BLUE), (rp, RED)]:
        # glow effect
        glow = pygame.Surface((rect.width + 14, rect.height + 14), pygame.SRCALPHA)
        pygame.draw.rect(glow, (*col, 35), glow.get_rect(), border_radius=7)
        screen.blit(glow, (rect.x - 7, rect.y - 7))
        pygame.draw.rect(screen, col, rect, border_radius=4)

    # scores
    if not infinite_mode:
        sl = fonts["huge"].render(str(rs["score_left"]), True, BLUE)
        sr = fonts["huge"].render(str(rs["score_right"]), True, RED)
        screen.blit(sl, (WIDTH // 4 - sl.get_width() // 2, 18))
        screen.blit(sr, (3 * WIDTH // 4 - sr.get_width() // 2, 18))

    # yellow rally counter
    rally_val = rs["rally"]
    if rally_val >= 30:
        rc = GREEN
    elif rally_val >= 15:
        rc = YELLOW
    else:
        rc = WHITE

    rl_surf = fonts["mid"].render(f"Rally: {rally_val}", True, rc)
    screen.blit(rl_surf, rl_surf.get_rect(center=(WIDTH // 2, HEIGHT - 22)))


def draw_panel(screen, fonts, left_agent, rally_label, infinite_mode):
    py = HEIGHT
    pygame.draw.rect(screen, PANEL_BG, (0, py, WIDTH, PANEL_HEIGHT))
    pygame.draw.line(screen, SEP_LINE, (0, py), (WIDTH, py), 2)

    lx = 20
    heading = fonts["mid"].render(rally_label, True, TEAL)
    screen.blit(heading, (lx, py + 12))

    if infinite_mode:
        mode_str = "infinite mode"
    else:
        mode_str = "training snapshot"

    screen.blit(fonts["small"].render(mode_str, True, LABEL_COL), (lx, py + 42))
    screen.blit(fonts["small"].render("SPACE pause  ·  R reset  ·  ESC quit", True, LABEL_COL), (lx, py + 62))

    # settings
    rx = WIDTH // 2 + 20

    settings = [
        ("learning Rate (α)", f"{ALPHA}"),
        ("discount (γ)", f"{GAMMA}"),
        ("ε min", f"{EPSILON_MIN}"),
        ("ε decay / rally", f"{EPSILON_DECAY}"),
        ("hit reward", f"+{REWARD_HIT}"),
        ("miss penalty", f"{REWARD_MISS}"),
    ]

    col_w = (WIDTH - rx - 20) // 2
    row_h = (PANEL_HEIGHT - 16) // 3

    for i, (lbl, val) in enumerate(settings):
        col = i % 3
        row = i // 3
        x = rx + col * (col_w + 12)
        y = py + 10 + row * row_h
        screen.blit(fonts["label"].render(lbl, True, LABEL_COL), (x, y))
        screen.blit(fonts["small"].render(val, True, WHITE), (x, y + 16))


def run_visualiser(left_path, right_path, rally_label, infinite_mode=False, window_title="Pong – Q-Learning"):
    pygame.init()
    pygame.display.set_caption(window_title)
    screen = pygame.display.set_mode((WIDTH, HEIGHT + PANEL_HEIGHT))
    clock = pygame.time.Clock()

    fonts = {
        "huge":  pygame.font.Font(None, 80),
        "big":   pygame.font.Font(None, 48),
        "mid":   pygame.font.Font(None, 30),
        "small": pygame.font.Font(None, 22),
        "label": pygame.font.Font(None, 19),
    }

    left_agent = QLearningAgent("Left")
    right_agent = QLearningAgent("Right")
    left_agent.load(left_path)
    right_agent.load(right_path)

    # no exploration
    left_agent.epsilon = 0.0
    right_agent.epsilon = 0.0

    env = PongEnv()
    state_l, state_r = env.reset()
    trail = []
    paused = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    state_l, state_r = env.reset()
                    trail.clear()

        if paused:
            screen.fill(BLACK)
            lbl = fonts["big"].render("PAUSED  –  SPACE to resume", True, WHITE)
            screen.blit(lbl, lbl.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
            pygame.display.flip()
            clock.tick(15)
            continue

        action_l = left_agent.choose_action(state_l)
        action_r = right_agent.choose_action(state_r)
        next_l, next_r, _, _, rally_ended = env.step(action_l, action_r)
        state_l, state_r = next_l, next_r

        rs = env.render_state

        # update trail
        trail.append((int(rs["ball_x"]), int(rs["ball_y"]), MAX_TRAIL))
        trail = [(x, y, a - 1) for x, y, a in trail if a > 0]
        if len(trail) > MAX_TRAIL:
            trail = trail[-MAX_TRAIL:]

        draw_game(screen, rs, trail, fonts, infinite_mode=infinite_mode)
        draw_panel(screen, fonts, left_agent, rally_label, infinite_mode)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()