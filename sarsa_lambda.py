# Sai Kiran Maddela, Vivek Ramanathan
# CS394R - Reinforcement Learning Final Project

from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import random
import pickle as pkl

# State space constants
max_vert = 300
min_vert = -150
max_horiz = 400
min_horiz = 0
max_vel = 20
min_vel = -20
num_actions = 2

# Binning constants
vert_bin = 2
horiz_bin = 2
vel_bin = 2
feature_len = int((max_vert - min_vert) / vert_bin) * int((max_horiz - min_horiz) / horiz_bin) * int((max_vel - min_vel) / vel_bin)

# Returns feature vector for a give state, done, action
def get_feature_vector(state, done, action):
    x = np.zeros(num_actions * feature_len)
    if done:
        return x
    
    start = int(action * feature_len / num_actions)
    v, h, vel = state
    v_idx, h_idx, vel_idx = (v - min_vert) / vert_bin, (h - min_horiz) / horiz_bin, (vel - min_vel) / vel_bin
    x[start + int(v_idx)] = 1
    x[start + int((max_vert - min_vert) / vert_bin) + int(h_idx)] = 1
    x[start + int((max_vert - min_vert) / vert_bin) + int((max_horiz - min_horiz) / horiz_bin) + int(vel_idx)] = 1
    
    return x

# Use epsilon-greedy policy and weight vector to get action
def choose_action(S, done, w):
    actions = [0, 119]
    prob = random.uniform(0, 1)
    Q = [np.dot(w, get_feature_vector(S, done, 0)), np.dot(w, get_feature_vector(S, done, 1))]
    if prob <= epsilon:
        action = random.randint(0, 1)
    else:
        if Q[0] > Q[1]:
            action = 0
        elif Q[0] < Q[1]:
            action = 1
        else:
            action = random.randint(0, 1)
    
    return actions[action], action


game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
p.init()

# Algorithm specific constants
alpha = 0.1
epsilon = 0.0
num_episodes = 100000
gamma = 0.90
lamb = 0.8

iteration = 0

max_score = 0 
game.adjustRewards({'positive' : 5.0, 'loss' : -1000, 'tick' : 0.5})
w = np.zeros(2 * feature_len)

mean_s = []
max_s = []
scores = []

# Follows pseudocode provided in Sutton Book
for episode in range(num_episodes):
    # Observe current state
    S = game.getGameState()
    vertical_dist = int(S['next_pipe_bottom_y'] - S['player_y'])
    horizontal_dist = int(S['next_pipe_dist_to_player'])
    velocity = int(S['player_vel'])
    action, idx = choose_action((vertical_dist, horizontal_dist, velocity), False, w)
    
    x = get_feature_vector((vertical_dist, horizontal_dist, velocity), False, idx)
    z = np.zeros(2 * feature_len)
    Q_old = 0
    curr_score = 0
    while not p.game_over():
        max_score = max(max_score, game.pipes_passed)
        curr_score = max(curr_score, game.pipes_passed)
        iteration += 1

        # Action action and observe next state
        R = p.act(action)
        S_prime = game.getGameState()
        vertical_dist = int(S_prime['next_pipe_bottom_y'] - S_prime['player_y'])
        horizontal_dist = int(S_prime['next_pipe_dist_to_player'])
        velocity = int(S_prime['player_vel'])
        action, idx = choose_action((vertical_dist, horizontal_dist, velocity), p.game_over(), w)

        # Compute vectors and eligibility trace computation
        x_prime = get_feature_vector((vertical_dist, horizontal_dist, velocity), p.game_over(), idx)
        Q = np.dot(w, x)
        Q_prime = np.dot(w, x_prime)
        delta = R + gamma * Q_prime - Q
        z = lamb * gamma * z + (1 - alpha * gamma * lamb * np.matmul(np.transpose(z), x)) * x
        w = w + alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
        Q_old = Q_prime
        x = x_prime

    scores.append(curr_score)
    mean_s.append(sum(scores) / len(scores))
    max_s.append(max_score)

    # Store pickle files
    if episode % 1000 == 0:
        with open('sarsa_l_w.pkl','wb') as f: pkl.dump(w, f)
        with open('sarsa_l_max.pkl','wb') as f: pkl.dump(max_s, f)
        with open('sarsa_l_mean.pkl','wb') as f: pkl.dump(mean_s, f)


    print("Game", episode, "Current: ", curr_score, "Max: ", max_score)
    p.reset_game()
