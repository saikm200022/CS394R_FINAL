# Sai Kiran Maddela, Vivek Ramanathan
# CS394R - Reinforcement Learning Final Project

from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import random
import pickle as pkl

# epsilon-greedy policy
def choose_action(s0, s1, s2, Q1, Q2):
    actions = [0, 119]
    prob = random.uniform(0, 1)
    if prob <= epsilon:
        action = random.randint(0, 1)
    else:
        if (Q1[s0, s1, s2, 0] + Q2[s0, s1, s2, 0]) / 2 > (Q1[s0, s1, s2, 1] + Q2[s0, s1, s2, 1]) / 2:
            action = 0
        elif (Q1[s0, s1, s2, 0] + Q2[s0, s1, s2, 0]) / 2 < (Q1[s0, s1, s2, 1] + Q2[s0, s1, s2, 1]) / 2:
            action = 1
        else:
            action = random.randint(0, 1)
    
    return actions[action], action

# Relevant algorithm constants
alpha = 0.1
epsilon = 0.0
gamma = 0.90

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
p.init()

# State space constants
max_vert = 300
min_vert = -150
max_horiz = 400
min_horiz = 0
max_vel = 20
min_vel = -20

iteration = 0

# Initialize both tables
Q1 = np.random.rand(max_vert - min_vert, max_horiz - min_horiz, max_vel - min_vel, 2)
Q2 = np.random.rand(max_vert - min_vert, max_horiz - min_horiz, max_vel - min_vel, 2)

# Reward system
game.adjustRewards( {'positive' : 5.0, 'loss' : -1000, 'tick' : 0.5})
max_score = 0
game_id = 0

scores = []
mean_score = []
max_s = []

# Follows pseudocode provided in Sutton book
for episode in range(100000):
    game_score = 0
    while not p.game_over():
        max_score = max(max_score, game.pipes_passed)
        game_score = max(game_score, game.pipes_passed)
        iteration += 1

        # Get current state and convert into index format to access Q Table
        state = game.getGameState()
        vertical_dist = int(state['next_pipe_bottom_y'] - state['player_y'])
        horizontal_dist = int(state['next_pipe_dist_to_player'])
        velocity = int(state['player_vel'])

        v_idx = vertical_dist - min_vert
        h_idx = horizontal_dist
        vel_idx = velocity - min_vel

        action, idx = choose_action(v_idx, h_idx, vel_idx, Q1, Q2)

        # Act action and observe next state
        reward = p.act(action)
        next_state = game.getGameState()
        next_state_vertical_dist = int(next_state['next_pipe_bottom_y'] - next_state['player_y'])
        next_horizontal_dist = int(next_state['next_pipe_dist_to_player'])
        next_velocity = int(next_state['player_vel'])
        
        # With prob < 0.5 update 1 Q table otherwise update the other
        if random.uniform(0, 1) < 0.5:
            bootstrap = 0
            if not p.game_over():
                a = np.argmax(Q1[next_state_vertical_dist - min_vert, next_horizontal_dist - min_horiz, next_velocity - min_vel])
                bootstrap = Q2[next_state_vertical_dist - min_vert, next_horizontal_dist - min_horiz, next_velocity - min_vel, int(a)]
            Q1[v_idx, h_idx, vel_idx, idx] += alpha * (reward + gamma * bootstrap - Q1[v_idx, h_idx, vel_idx, idx])
        
        else:
            bootstrap = 0
            if not p.game_over():
                a = np.argmax(Q2[next_state_vertical_dist - min_vert, next_horizontal_dist - min_horiz, next_velocity - min_vel])
                bootstrap = Q1[next_state_vertical_dist - min_vert, next_horizontal_dist - min_horiz, next_velocity - min_vel, int(a)]
            Q2[v_idx, h_idx, vel_idx, idx] += alpha * (reward + gamma * bootstrap - Q2[v_idx, h_idx, vel_idx, idx])
    
    scores.append(game_score)
    mean_score.append(sum(scores) / len(scores))
    max_s.append(max(scores))
    p.reset_game()
    game_id += 1

    # Code to save progress
    if episode % 10000 == 0:
        with open('double_q_Q1_avg.pkl','wb') as f: pkl.dump(Q1, f)
        with open('double_q_Q2_avg.pkl','wb') as f: pkl.dump(Q2, f)
        with open('double_q_max_avg.pkl','wb') as f: pkl.dump(max_s, f)
        with open('double_q_mean_avg.pkl','wb') as f: pkl.dump(mean_score, f)
    
    print("GAME: ", game_id, "Score: ", game_score, "Max: ", max_score)