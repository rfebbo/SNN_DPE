import numpy as np
import matplotlib.pyplot as plt

from encoder import *
from network_tools import *
from iris_data_tools import *
from training_tools import *
from neuron_synapse import *
from plotting_tools import *

def encoder_test():
    min_f = 10
    max_f = 100
    sim_f = 1000

    value1 = 0.0
    value2 = 0.1
    value3 = 1.0

    e1 = Encoder(min_f, max_f, sim_f) 
    e2 = Encoder(min_f, max_f, sim_f)
    e3 = Encoder(min_f, max_f, sim_f)
    
    e1.set_value(value1)
    e2.set_value(value2)
    e3.set_value(value3)

    # simulate for one second
    fires1 = []
    fires2 = []
    fires3 = []
    for t in range(sim_f):
        if e1.update():
            fires1.append(t)

        if e2.update():
            fires2.append(t)

        if e3.update():
            fires3.append(t)

    fig = plt.subplots(figsize=(13,2))
    plt.title(f'Encoder Outputs with a frequency range of {min_f} Hz - {max_f} Hz simulated at {sim_f} Hz for 1 second')
    e_labels=[f'Value1: {value1} -> {e1.value_f} Hz',f'Value2: {value2} -> {e2.value_f} Hz',f' Value3: {value3} -> {e3.value_f} Hz']
    plt.yticks(ticks=[1, 2, 3], labels=e_labels)
    plt.ylabel('Encoders')
    plt.ylim(0.5, 3.5)
    plt.xlabel('Time (ms)')

    plt.xticks(ticks=range(0, sim_f + 1, 100))
    
    plt.scatter(fires1, np.ones(len(fires1)) * 1, marker='|')
    plt.scatter(fires2, np.ones(len(fires2)) * 2, marker='|')
    plt.scatter(fires3, np.ones(len(fires3)) * 3, marker='|')
    # plt.savefig('Images/Example_enc.pdf')
    plt.show()

def run_network_test(normalized_iris_data_sample, attributes):
    # create a test network and encoders
    n_neurons = 16
    n_synapses = int(n_neurons * np.random.uniform(low=2, high=3)) # random number from n_neurons * 2 to n_neurons * 3

    neurons = create_network(n_neurons, n_synapses)

    encoders = create_encoders(attributes)

    sim_time = 200

    # feed a test sample into the test network
    fires = run_network(neurons, encoders, normalized_iris_data_sample, sim_time)

    reset_network(neurons, encoders)

    plot_spikes(fires, attributes, normalized_iris_data_sample, sim_time)

    # visualize network
    G = nx.Graph()

    for n in neurons:
        G.add_node(n.id)

        for s in n.synapses:
            G.add_edge(n.id, s.n2.id)

    nx.draw(G)

def steady_state_test(normalized_iris_data_sample, attributes):
    # create a test network and encoders
    n_neurons = 16
    n_synapses = int(n_neurons * np.random.uniform(low=2, high=3)) # random number from n_neurons * 2 to n_neurons * 3

    neurons = create_network(n_neurons, n_synapses)

    encoders = create_encoders(attributes)

    sim_time = 200
    # feed a test sample into the test network and run it
    fires = run_network(neurons, encoders, normalized_iris_data_sample, sim_time)

    reset_network(neurons, encoders)

    plot_spikes(fires, attributes, normalized_iris_data_sample, sim_time)

    window_size = 10
    fire_matrix, total_fires, steady_state_t, (m1, m2) = find_steady_state(sim_time, attributes, fires, window_size=window_size)

    if steady_state_t == 0:
        steady_state_t = sim_time
    
    # plotting code
    t = range(steady_state_t)
    plt.plot(t, total_fires)
    plt.xlabel('Time (ms)')
    plt.ylabel('Overall Network Spike Rate')
    plt.title('Spike Rate Over Time Until Steady State Reached')

    # add text and lines
    l_pos = [t[-1], t[-window_size-1], t[-window_size*2-1]] 
    tx1_pos = [l_pos[1] + window_size / 2, max(total_fires) / 2]
    tx2_pos = [l_pos[2] + window_size / 2, max(total_fires) / 2.5]
    plt.vlines(l_pos, 0, max(total_fires), colors='r', linestyles='dashed')
    plt.text(tx1_pos[0], tx1_pos[1], f'avg spikes: {m2}', horizontalalignment='center')
    plt.text(tx2_pos[0], tx2_pos[1], f'avg spikes: {m1}', horizontalalignment='center')
    # plt.savefig('Images/Spike_rate_steady_state.pdf')
    plt.show()

    fire_matrix = run_network_early_exit(neurons, encoders, normalized_iris_data_sample, sim_time, window_size=window_size)

    reset_network(neurons, encoders)

    plot_fire_matrix(fire_matrix)

def weight_update_test(normalized_iris_data_sample, attributes, label, classes):
    # create a test network and encoders
    n_neurons = 16
    n_synapses = int(n_neurons * np.random.uniform(low=2, high=3)) # random number from n_neurons * 2 to n_neurons * 3

    neurons = create_network(n_neurons, n_synapses)

    encoders = create_encoders(attributes)

    dpe_weights = np.random.rand(n_neurons, len(classes))

    sim_time = 200

    # show we can reduce error for a singe test sample
    for i in range(10):
        # feed a test sample into the test network
        fire_matrix = run_network_early_exit(neurons, encoders, normalized_iris_data_sample, sim_time)

        reset_network(neurons, encoders)

        # fire_matrix, total_fires, steady_state_t, _ = find_steady_state(sim_time, attributes, fires)

        x, y = forward_pass(fire_matrix, dpe_weights)

        y_hat = np.zeros(len(classes))
        y_hat[label] = 1

        print(f'E{i} = {mse(y, y_hat)}')

        update_weights(fire_matrix, dpe_weights, x, y, y_hat)