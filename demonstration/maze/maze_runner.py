import matplotlib.pyplot as plt
import numpy as np

import random


from snn_dpe.tools.data import normalize_iris_data, read_iris_data
from snn_dpe.tools.network import (create_encoders, create_network,
                                   reset_network, run_network)
from snn_dpe.tools.plotting import plot_network, plot_spike_raster, plot_spikes

from maze_gen import (add_pos, actions_to_dirs, create_maze,
                      dir_str, dirs_str, get_action_vec, get_maze_junction_ratio,
                      plot_maze, test_plot_maze, update_dir)
from snn_dpe.tools.train.utils import forward_pass, update_weights








if __name__ == "__main__":
    debug = False
    desired_junction_ratio = 0.05 # percentage of paths that have 3 or 4 exits

    # test_plot_maze()

    maze, start, start_dir = create_maze(desired_junction_ratio, debug=debug)

    junction_ratio = get_maze_junction_ratio(maze)
    print(f"Junction ratio: {junction_ratio}")

    cur_pos = start
    cur_dir = start_dir

    # create a test network and encoders
    n_neurons = 16
    n_synapses = 32

    neurons = create_network(n_neurons, n_synapses)

    min_f = 0
    max_f = 800
    sim_f = 1000

    sim_time = 200


    encoders = create_encoders(3, min_f, max_f, sim_f)

    dpe_weights = np.random.rand(n_neurons, 3) - 0.5
    dpe_bias = np.random.rand(n_neurons) * 0.1


    for _ in range(1000):
        # get one-hot action vector for the current position
        action_vec = get_action_vec(maze, cur_pos, cur_dir)
        print(f'cur_pos: {cur_pos}')
        print(f'heading: {dir_str(cur_dir)}')
        print(f'action vec: {action_vec}')
        
        # get list of possible actions in direction format (UP, DOWN, LEFT, RIGHT)
        available_directions = actions_to_dirs(action_vec, cur_dir)
        print(f'available_directions: {dirs_str(available_directions)}')

        plot_maze(maze, cur_pos, debug=True)

        
        # feed a test sample into the test network
        spike_raster, encoder_fires = run_network(neurons, encoders, action_vec, sim_time)

        reset_network(neurons, encoders)

        plot_spike_raster(spike_raster)

        x, y = forward_pass(spike_raster, dpe_weights, dpe_bias)
        action_idx = np.argmax(y)
        if action_vec[action_idx] == 0:
            print('Chose invalid action')
        else:
            # action = random.choice(available_directions)
            print(f'Action: {dir_str(action_idx)}')
            cur_pos = add_pos(cur_pos, action_idx)
            cur_dir = update_dir(maze, cur_pos, action_idx)