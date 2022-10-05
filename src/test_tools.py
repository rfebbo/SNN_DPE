import numpy as np

from network_tools import run_network_early_exit, reset_network
from training_tools import forward_pass


def predict(neurons, encoders, dpe_weights, sample, sim_time=100, window_size=10):
    # run network
    fire_matrix = run_network_early_exit(neurons, encoders, sample, sim_time, window_size=window_size)

    reset_network(neurons, encoders)

    _, y = forward_pass(fire_matrix, dpe_weights)

    return y
