import argparse
import csv
import os
import time

from agent import QLearningAgent
from game_env import PongEnv
from config import MODEL_LEFT, MODEL_RIGHT, CKPT_LEFT, CKPT_RIGHT, RALLY_CHECKPOINTS

LOG_FILE = "training_log.csv"


def train(target_rallies):
    env = PongEnv()
    left = QLearningAgent("Left")
    right = QLearningAgent("Right")

    left.load(MODEL_LEFT)
    right.load(MODEL_RIGHT)

    if target_rallies == 0:
        # print current stats
        print("Left agent: rallies =", left.total_rallies, "epsilon =", round(left.epsilon, 4), "states =", left.q_table_size)
        print("Right agent: rallies =", right.total_rallies, "epsilon =", round(right.epsilon, 4), "states =", right.q_table_size)
        return

    pending_ckpts = sorted([r for r in RALLY_CHECKPOINTS if r > left.total_rallies])

    start_rallies = left.total_rallies
    goal_rallies = start_rallies + target_rallies

    print("starting training...")
    print(f"Goal: {goal_rallies} total rallies")
    print(f"checkpoints to hit: {pending_ckpts}")

    state_l, state_r = env.reset()
    start_time = time.time()

    # keep track of rallys to get average
    rally_lens = []
    current_len = 0

    # set up csv log file
    log_exists = os.path.exists(LOG_FILE)
    log_f = open(LOG_FILE, "a", newline="")
    log_writer = csv.writer(log_f)
    if not log_exists:
        log_writer.writerow(["rally", "avg_rally_length", "epsilon", "left_states", "right_states"])

    while left.total_rallies < goal_rallies:
        action_l = left.choose_action(state_l)
        action_r = right.choose_action(state_r)

        next_l, next_r, rew_l, rew_r, rally_ended = env.step(action_l, action_r)

        # fix
        done = (left.total_rallies + 1) >= goal_rallies

        left.learn(state_l, action_l, rew_l, next_l, done)
        right.learn(state_r, action_r, rew_r, next_r, done)

        state_l = next_l
        state_r = next_r

        if rally_ended:
            left.on_rally_end()
            right.on_rally_end()
            rally_lens.append(current_len)
            # only keep last 200
            if len(rally_lens) > 200:
                rally_lens.pop(0)
            current_len = 0
        else:
            current_len = env.rally

        # save checkpoint if we hit one
        if pending_ckpts and left.total_rallies >= pending_ckpts[0]:
            ckpt = pending_ckpts.pop(0)
            print(f"\nsaving checkpoint at {ckpt} rallies")
            left.save(CKPT_LEFT.format(ckpt))
            right.save(CKPT_RIGHT.format(ckpt))

        # print progress and log to csv every 500 rallies
        if left.total_rallies % 500 == 0 and left.total_rallies > start_rallies:
            elapsed = time.time() - start_time
            done_so_far = left.total_rallies - start_rallies
            pct = done_so_far / target_rallies * 100
            
            if done_so_far > 0:
                eta = (elapsed / done_so_far) * (target_rallies - done_so_far)
            else:
                eta = 0
                
            if len(rally_lens) > 0:
                avg = sum(rally_lens) / len(rally_lens)
            else:
                avg = 0

            print(f"rally {left.total_rallies}/{goal_rallies} ({pct:.1f}%) - ETA: {eta:.0f}s - epsilon: {left.epsilon:.3f} - avg rally len: {avg:.1f}", end="\r")

            # write row to csv log
            log_writer.writerow([
                left.total_rallies,
                round(avg, 2),
                round(left.epsilon, 4),
                len(left.q_table),
                len(right.q_table)
            ])
            log_f.flush()

    log_f.close()

    print("\ndone!")
    left.save(MODEL_LEFT)
    right.save(MODEL_RIGHT)

    elapsed = time.time() - start_time
    avg = sum(rally_lens) / len(rally_lens) if rally_lens else 0

    print(f"\nfinished in {elapsed:.1f} seconds")
    print(f"left: {left.total_rallies} rallies, epsilon = {left.epsilon:.4f}, states = {left.q_table_size}")
    print(f"right: {right.total_rallies} rallies, epsilon = {right.epsilon:.4f}, states = {right.q_table_size}")
    print(f"average rally length (last 200): {avg:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rallies", "-r", type=int, default=50000, help="how many rallies to train for")
    args = parser.parse_args()
    train(args.rallies)