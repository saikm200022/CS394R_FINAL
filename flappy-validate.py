from ple.games.flappybird import FlappyBird
from ple import PLE

import pickle as pkl

# import argparse
import sys

max_vert = 300
min_vert = -150
max_horiz = 400
min_horiz = 0
max_vel = 20
min_vel = -20

def choose_action(s0, s1, s2, Q):
    actions = [0, 119]
    if Q[s0, s1, s2][0] > Q[s0, s1, s2][1]:
        action = 0
    elif Q[s0, s1, s2][0] < Q[s0, s1, s2][1]:
        action = 1
    else:
        action = random.randint(0, 1)

    return actions[action], action


def main(Q_filename):

    game = FlappyBird()
    p = PLE(game, fps=30, display_screen=True)
    p.init()

    file = open(Q_filename,'rb') # change Q.pkl to point to your Q matrix
    Q = pkl.load(file)

    # validation
    num_episodes = 100
    episode = 0

    pipes_passed = [0]*num_episodes

    while episode < num_episodes:
        while not p.game_over():
            pipes_passed[episode] = game.pipes_passed
            state = game.getGameState()
            vertical_dist = int(state['next_pipe_bottom_y'] - state['player_y'])
            horizontal_dist = int(state['next_pipe_dist_to_player'])
            velocity = int(state['player_vel'])

            v_idx = vertical_dist - min_vert
            h_idx = horizontal_dist
            vel_idx = velocity - min_vel

            action, idx = choose_action(v_idx, h_idx, vel_idx, Q)

            p.act(action)

        # pipes_passed.append(game.pipes_passed)
        p.reset_game()
        episode += 1

    print("avg pipes passed=",sum(pipes_passed)/num_episodes)
    print("max pipes passed=",max(pipes_passed))



if __name__ == "__main__":
    argv = sys.argv
    if len(argv) <= 1:
        raise Exception("Not enough arguments provided!")
    filename = argv[1]
    main(filename)
