import numpy as np
import json

from neuron_synapse import Neuron, Synapse
from encoder import Encoder

def create_network(n_neurons, n_synapses, negative_weights = False, threshold_range = (0.35, 0.55), leak_range = (0.05, 0.25), weight_factor = 1.8):
    Neurons = []

    for i in range(n_neurons):
        threshold = np.random.uniform(low=threshold_range[0], high=threshold_range[1]) 
        leak = np.random.uniform(low=leak_range[0], high=leak_range[1]) 
        n = Neuron(i, threshold, leak)
        Neurons.append(n)

    for i in range(n_synapses):
        n1_id = np.random.choice(range(n_neurons))
        n2_id = np.random.choice(range(n_neurons))

        n1 = Neurons[n1_id]
        n2 = Neurons[n2_id]

        # positive weights only (should we keep it like that?)
        weight = np.random.rand(1) * weight_factor

        if negative_weights:
            weight -= weight_factor/2

        s = Synapse(n1, n2, weight)

        Neurons[n1_id].add_synapse(s)

    return Neurons

# create a network with encoders and feed it a sample from the dataset
def create_encoders(attributes, min_f = 10, max_f = 700, sim_f = 1000):
    encoders = []
    for i, a in attributes.items():
        e = Encoder(min_f, max_f, sim_f)
        encoders.append(e)

    return encoders

def run_network(neurons, encoders, enc_input, sim_time):

    for i, e in enumerate(encoders):
        e.set_value(enc_input[i])

    # simulate
    fires = []
    for i in range(len(neurons) + len(encoders)):
        fires.append([])

    for t in range(sim_time):
        # get the input for this timestep, and apply it to input neurons
        for i, e in enumerate(encoders):
            if e.update():
                fires[i].append(t)
                neurons[i].apply_potential(1)

        # update the network
        for n in neurons:
            if n.update():
                fires[n.id + len(encoders)].append(t)

    return fires

def reset_network(neurons, encoders):
    for n in neurons:
        n.reset()
    
    for e in encoders:
        e.reset()

# for each timestep
#   sum the total fires
#   if the average fires over the last window_size timesteps is about equal to the windowsize before that, steady state has been reached
def find_steady_state(sim_time, attributes, fires, window_size=10):
    # will hold the output of each neuron over the steady state (0 -> no fire @ t, 1 -> fire @ t)
    fire_matrix = [] 
    # the sum of all fires over the simulation time
    total_fires = []
    steady_state_t = 0
    m1 = 0
    m2 = 0

    for t in range(sim_time):
        fire_matrix.append([])
        fires_at_t = 0

        # exclude encoder spikes
        for f in fires[len(attributes):]:
            
            if t in f:
                fires_at_t += 1
                fire_matrix[-1].append(1)
            else:
                fire_matrix[-1].append(0)

        if t > window_size*2:
            m1 = np.mean(total_fires[-window_size*2:-window_size])
            m2 = np.mean(total_fires[-window_size:])
            if np.isclose(m1, m2):
                # print(f'steady state at {t}')
                steady_state_t = t
                break

        total_fires.append(fires_at_t)

    return np.asarray(fire_matrix), total_fires, steady_state_t, (m1, m2)

# an updated run_network function with early exit condition determined by steady state logic
def run_network_early_exit(neurons, encoders, enc_input, sim_time, window_size=10):
    # for determing if steady state reached (see next cell)
    total_fires = []
    for i, e in enumerate(encoders):
        e.set_value(enc_input[i])

    # simulate
    fire_matrix = []

    for t in range(sim_time):
        fire_matrix.append([])
        total_fires.append(0)
        # get the input for this timestep, and apply it to input neurons
        for i, e in enumerate(encoders):
            if e.update():
                neurons[i].apply_potential(1)

        # update the network
        for n in neurons:
            if n.update():
                fire_matrix[-1].append(1)
                total_fires[-1] += 1
            else:
                fire_matrix[-1].append(0)

        if t > window_size*2:
            m1 = np.mean(total_fires[-window_size*2:-window_size])
            m2 = np.mean(total_fires[-window_size:])
            if np.isclose(m1, m2):
                # print(f'steady state at {t}')
                steady_state_t = t
                break

    return np.asarray(fire_matrix)

def save_trained_network(filename, neurons, encoders, dpe_weights, window_size, c_acc, E_t, avg_ss):
    n_neurons = len(neurons)

    network = {}

    network['neurons'] = {}
    network['encoders'] = {}
    network['dpe_weights'] = dpe_weights
    network['window_size'] = window_size
    network['cumulative accuracy'] = c_acc
    network['Epoch Accuracy'] = E_t
    network['Average Steady State Time'] = avg_ss

    for n in neurons:
        network['neurons'][n.id] = {}
        network['neurons'][n.id]['synapses'] = []
        network['neurons'][n.id]['leak'] = n.leak
        network['neurons'][n.id]['threshold'] = n.threshold

        for s in n.synapses:
            syn = {}
            syn['n1'] = s.n1
            syn['n2'] = s.n2
            syn['weight'] = s.weight
            network['neurons'][n.id]['synapses'].append(syn)

    for i, e in enumerate(encoders):
        network['encoders'][i] = {}
        network['encoders'][i] = {}
