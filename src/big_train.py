import numpy as np
import matplotlib.pyplot as plt
import copy
from multiprocessing import Pool

import sys
sys.path.append('../src')

from network_tools import create_network, create_encoders
from neuron_synapse import Neuron, Synapse
from network_tools import reset_network, run_network
from iris_data_tools import read_iris_data, normalize_iris_data
from training_tools import train_all
from plotting_tools import plot_acc, plot_network

iris_data_location = '../Data/Iris/iris.data'

def dpe_conv_search(args):
    normalized_iris_data = args[0]
    encoders = args[1]
    classes = args[2]
    labels = args[3]

    sim_time = args[4]
    window_size = args[5]
    n_epochs = args[6]

    #  create network and encoders
    n_neurons = 16
    n_synapses = int(n_neurons * np.random.uniform(low=2, high=3)) # random number from n_neurons * 2 to n_neurons * 3

    
    neurons = create_network(n_neurons, n_synapses)
    dpe_weights = np.random.rand(n_neurons, len(classes))
    E_t, avg_ss, c_acc = train_all(normalized_iris_data, labels, classes, neurons, encoders, dpe_weights, sim_time=sim_time, window_size=window_size, n_epochs=n_epochs)


    output_neurons = []
    best_copy = copy.deepcopy(neurons)

    for i in range(3):
        n = Neuron(i+4, 0.5, 0.0)

        for j in range(4):
            # if best_weights[j][i] > 0:
            n1 = best_copy[j]
            n2 = n

            s = Synapse(n1, n2, dpe_weights[j][i])
            n1.add_synapse(s)

        output_neurons.append(n)

    reset_network(neurons, encoders)

    best_copy += output_neurons
    # plot_network(best_copy)

    # for n in best_copy:
        # print(n.id)

    sample = 6
    n_correct = 0
    for sample, label in zip(range(len(normalized_iris_data)),labels):
        # feed a test sample into the test network
        fires = run_network(best_copy, encoders, normalized_iris_data[sample], sim_time)

        # plot_spikes(fires, attributes, normalized_iris_data[sample], sim_time)

        sums = []
        for f in fires:
            sums.append(np.sum(len(f)))

        # for e in encoders:
        #     print(int(100/e.fire_period))

        # print(sums[-3:])
        pred = np.argmax(sums[-3:])
        if pred == label:
            n_correct += 1

    return (n_correct, best_copy, dpe_weights)

iris_data, labels, classes, attributes = read_iris_data(iris_data_location, shuffle=True)

normalized_iris_data = normalize_iris_data(iris_data, attributes)

encoders = create_encoders(attributes)
best_neurons = None
best_weights = None
best_E_t = None
best_avg_ss = None
best_c_acc = None
max = 0

n_epochs = 10
window_size = 10
sim_time = 100

args = []
for n in range(500):
    args.append((normalized_iris_data, encoders, classes, labels, sim_time, window_size, n_epochs))
    
for n in range(500):
    with Pool() as p:
        res = p.map(dpe_conv_search, args)

    for r in res:
        if r[0] > max:
            max = r[0]
            best_neurons = r[1]
            best_weights = r[2]

    max = float(max)/len(normalized_iris_data)
    print(max)