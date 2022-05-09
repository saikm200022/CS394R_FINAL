from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import random

import pickle as pkl
import sys
import csv

def choose_action(s0, s1, s2, Q):
    actions = [0, 119]
    prob = random.uniform(0, 1)
    if prob <= epsilon:
        action = random.randint(0, 1)
    else:
        if Q[s0, s1, s2][0] > Q[s0, s1, s2][1]:
            action = 0
        elif Q[s0, s1, s2][0] < Q[s0, s1, s2][1]:
            action = 1
        else:
            action = random.randint(0, 1)

    return actions[action], action

max_vert = 300
min_vert = -150
max_horiz = 400
min_horiz = 0
max_vel = 20
min_vel = -20

alpha = 0.1
epsilon = 0.00
gamma = 0.90

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
p.init()
game.adjustRewards({'positive' : 5.0, 'loss' : -1000, 'tick' : 0.5})

num_episodes = 100000
episode = 0
pipes_passed = [0]*num_episodes

Q = np.random.rand(max_vert - min_vert, max_horiz - min_horiz, max_vel - min_vel, 2)
max_score = 0

while episode < num_episodes:
    while not p.game_over():
        pipes_passed[episode] = game.pipes_passed
        max_score = max(max_score, game.pipes_passed)


        state = game.getGameState()
        vertical_dist = int(state['next_pipe_bottom_y'] - state['player_y'])
        horizontal_dist = int(state['next_pipe_dist_to_player'])
        velocity = int(state['player_vel'])

        v_idx = vertical_dist - min_vert
        h_idx = horizontal_dist
        vel_idx = velocity - min_vel

        action, idx = choose_action(v_idx, h_idx, vel_idx, Q)

        reward = p.act(action)
        next_state = game.getGameState()
        next_state_vertical_dist = int(next_state['next_pipe_bottom_y'] - next_state['player_y'])
        next_horizontal_dist = int(next_state['next_pipe_dist_to_player'])
        next_velocity = int(next_state['player_vel'])

        next_state_Q = [0, 0]
        if not p.game_over():
            next_state_Q = Q[next_state_vertical_dist - min_vert, next_horizontal_dist, next_velocity - min_vel]
        Q[v_idx, h_idx, vel_idx, idx] += alpha * (reward + gamma * max(next_state_Q[0], next_state_Q[1]) - Q[v_idx, h_idx, vel_idx, idx])
    # Provide update to terminal with percentage done and max score
    if episode % 100 == 0:
        sys.stdout.write("\r%d%%" % int((episode/num_episodes)*100))
        sys.stdout.write(", max_score=%d" % max_score)
        sys.stdout.flush()

    episode += 1
    p.reset_game()

csv_file = 'csvs/qlearning_learning.csv'
pkl_file = 'pkls/qlearning.pkl'

# save learning progress for graphing later
with open(csv_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['episode','pipes_passed'])
    for i, pipes in enumerate(pipes_passed):
        writer.writerow([i,pipes])

# save Q
with open(pkl_file,'wb') as f:
    pkl.dump(Q, f)

