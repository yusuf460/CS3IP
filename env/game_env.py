import math
import random

from config import (
    WIDTH, HEIGHT,
    PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_SPEED,
    BALL_SIZE, BALL_SPEED,
    N_BALL_X, N_BALL_Y, N_BALL_VX, N_BALL_VY, N_PADDLE_Y,
    REWARD_HIT, REWARD_MISS,
)

LEFT_X = 30
RIGHT_X = WIDTH - 30 - PADDLE_WIDTH


class PongEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.left_y = (HEIGHT - PADDLE_HEIGHT) // 2
        self.right_y = (HEIGHT - PADDLE_HEIGHT) // 2
        self._launch_ball()
        self.rally = 0
        self.total_rallies = 0
        self.score_left = 0
        self.score_right = 0
        return self._state_l(), self._state_r()

    def step(self, action_l, action_r):
        # move paddles
        self.left_y = max(0, min(HEIGHT - PADDLE_HEIGHT, self.left_y + self._dy(action_l)))
        self.right_y = max(0, min(HEIGHT - PADDLE_HEIGHT, self.right_y + self._dy(action_r)))

        # move ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # bounce off top/bottom
        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y >= HEIGHT - BALL_SIZE:
            self.ball_y = HEIGHT - BALL_SIZE
            self.ball_vy = -abs(self.ball_vy)

        reward_l = reward_r = 0.0
        rally_ended = False

        ball_cx = self.ball_x + BALL_SIZE / 2
        ball_cy = self.ball_y + BALL_SIZE / 2

        # left paddle hit
        if (self.ball_vx < 0
                and LEFT_X <= ball_cx <= LEFT_X + PADDLE_WIDTH + abs(self.ball_vx) + 2
                and self.left_y - BALL_SIZE <= ball_cy <= self.left_y + PADDLE_HEIGHT + BALL_SIZE):
            self.ball_vx = abs(self.ball_vx) * 1.02
            offset = (ball_cy - (self.left_y + PADDLE_HEIGHT / 2)) / (PADDLE_HEIGHT / 2)
            self.ball_vy = max(-BALL_SPEED * 1.5, min(BALL_SPEED * 1.5, offset * BALL_SPEED * 1.2))
            reward_l += REWARD_HIT
            self.rally += 1
            self.total_rallies += 1

        # right paddle hit
        elif (self.ball_vx > 0
                and RIGHT_X - abs(self.ball_vx) - 2 <= ball_cx <= RIGHT_X + PADDLE_WIDTH
                and self.right_y - BALL_SIZE <= ball_cy <= self.right_y + PADDLE_HEIGHT + BALL_SIZE):
            self.ball_vx = -abs(self.ball_vx) * 1.02
            offset = (ball_cy - (self.right_y + PADDLE_HEIGHT / 2)) / (PADDLE_HEIGHT / 2)
            self.ball_vy = max(-BALL_SPEED * 1.5, min(BALL_SPEED * 1.5, offset * BALL_SPEED * 1.2))
            reward_r += REWARD_HIT
            self.rally += 1
            self.total_rallies += 1

        # check if someone scored
        if self.ball_x < -BALL_SIZE:
            self.score_right += 1
            reward_l += REWARD_MISS
            self.rally = 0
            self._launch_ball(towards="left")
            rally_ended = True
        elif self.ball_x > WIDTH:
            self.score_left += 1
            reward_r += REWARD_MISS
            self.rally = 0
            self._launch_ball(towards="right")
            rally_ended = True

        return self._state_l(), self._state_r(), reward_l, reward_r, rally_ended

    def _state_l(self):
        return (
            self._bucket(self.ball_x, 0, WIDTH, N_BALL_X),
            self._bucket(self.ball_y, 0, HEIGHT, N_BALL_Y),
            0 if self.ball_vx < 0 else 1,
            self._bucket(self.ball_vy, -BALL_SPEED * 1.5, BALL_SPEED * 1.5, N_BALL_VY),
            self._bucket(self.left_y, 0, HEIGHT - PADDLE_HEIGHT, N_PADDLE_Y),
        )

    def _state_r(self):
        return (
            self._bucket(WIDTH - self.ball_x, 0, WIDTH, N_BALL_X),
            self._bucket(self.ball_y, 0, HEIGHT, N_BALL_Y),
            0 if self.ball_vx > 0 else 1,
            self._bucket(-self.ball_vy, -BALL_SPEED * 1.5, BALL_SPEED * 1.5, N_BALL_VY),
            self._bucket(self.right_y, 0, HEIGHT - PADDLE_HEIGHT, N_PADDLE_Y),
        )

    def _launch_ball(self, towards=None):
        self.ball_x = WIDTH // 2 - BALL_SIZE // 2
        self.ball_y = HEIGHT // 2 - BALL_SIZE // 2
        sign = {"left": -1, "right": 1}.get(towards, random.choice([-1, 1]))
        angle = random.uniform(-0.6, 0.6)
        self.ball_vx = sign * BALL_SPEED * math.cos(angle)
        self.ball_vy = BALL_SPEED * math.sin(angle)

    @staticmethod
    def _dy(action):
        return {0: -PADDLE_SPEED, 1: 0, 2: PADDLE_SPEED}[action]

    @staticmethod
    def _bucket(value, lo, hi, n):
        r = max(0.0, min(1.0, (value - lo) / (hi - lo)))
        return min(int(r * n), n - 1)

    @property
    def render_state(self):
        return {
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "left_y": self.left_y,
            "right_y": self.right_y,
            "score_left": self.score_left,
            "score_right": self.score_right,
            "rally": self.rally,
            "total_rallies": self.total_rallies,
        }