import matplotlib.pyplot as plt
import numpy as np

from snn_dpe.tools.network import (create_encoders, create_network,
                                   reset_network, run_network)
from snn_dpe.tools.plotting import plot_network, plot_spike_raster
from sympy.utilities.iterables import multiset_permutations
import copy
import time
from datetime import datetime
import json
from matplotlib.lines import Line2D

import tqdm
from multiprocessing import Pool


def distance(a, b):
    total_distance = 0
    for n1, n2 in zip(a, b):
        for f1, f2 in zip(n1,n2):
            if f1 != f2:
                total_distance += 1

    return total_distance

def run_test(connectivity):
    # create a test network and encoders
    n_neurons = 100
    n_synapses = int((n_neurons ** 2) * connectivity)

    sim_time = 100

    min_f = 1
    max_f = 100
    sim_f = 100

    n_enc = 5

    network_inputs = []
    network_responses = []
    n_tests = 40

    run = {}
    run['n_neurons'] = n_neurons
    run['n_synapses'] = n_synapses
    run['connectivity'] = connectivity
    run['sim_time'] = sim_time
    run['min_f'] = min_f
    run['max_f'] = max_f
    run['sim_f'] = sim_f
    run['n_enc'] = n_enc
    run['n_tests'] = n_tests
    run['distances'] = {}
    run['saturations'] = {}

    for _ in tqdm.tqdm(range(n_tests)):
        np.random.seed()
        neurons = create_network(n_neurons, n_synapses, threshold_range=(1, 1), leak_range=(0.0, 0.0), weight_range=(1,1))

        encoders = create_encoders(n_enc, min_f, max_f, sim_f, enc_type='period')

        enc_inputs = np.random.uniform(0, 1, (n_enc))

        for enc_input in multiset_permutations(enc_inputs):
            fires = run_network(neurons, encoders, enc_input, sim_time)

            reset_network(neurons, encoders)

            network_responses.append(fires[:, n_enc:])
            network_inputs.append(fires[:, :n_enc])

        for i in range(len(network_inputs)):
            for j in range(i + 1, len(network_inputs)):
                input_d = distance(network_inputs[i], network_inputs[j]) /network_inputs[i].size
                response_d = distance(network_responses[i], network_responses[j]) /network_responses[i].size

                if input_d not in run['distances']:
                    run['distances'][input_d] = []
                    run['saturations'][input_d] = []

                run['distances'][input_d].append(response_d)
                run['saturations'][input_d].append(response_d)

    with open(f'./runs/{datetime.fromtimestamp(time.time())}', 'w') as f:
        f.write(json.dumps(run))


if __name__ == '__main__':
    with Pool(8) as p:
        points = [0.01, 0.02, 0.03, 0.99, 0.1, 0.11, 0.198, 0.199, 0.2]
        list(tqdm.tqdm(p.imap(run_test, points), total=len(points)))