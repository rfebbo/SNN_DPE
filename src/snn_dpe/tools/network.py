import json

import numpy as np

from snn_dpe import Encoder, Neuron, Synapse


def create_network(n_neurons, n_synapses, negative_weights = False, threshold_range = (0.35, 0.55), leak_range = (0.01, 0.25), weight_factor = 1, delay_range = (0, 4)):
    Neurons = []

    for i in range(n_neurons):
        threshold = np.random.uniform(low=threshold_range[0], high=threshold_range[1]) 
        leak = np.random.uniform(low=leak_range[0], high=leak_range[1]) 
        n = Neuron(threshold, leak)
        Neurons.append(n)

    for i in range(n_synapses):
        n1_id = np.random.choice(range(n_neurons))
        n2_id = np.random.choice(range(n_neurons))

        n1 = Neurons[n1_id]
        n2 = Neurons[n2_id]
        
        weight = np.random.rand(1)[0] * weight_factor

        if negative_weights:
            weight -= weight_factor/2

        delay = np.random.randint(delay_range[0], delay_range[1])

        s = Synapse(n1, n2, weight, delay)

        Neurons[n1_id].add_synapse(s)

    return Neurons

# create a network with encoders and feed it a sample from the dataset
def create_encoders(n_enc, min_f = 10, max_f = 700, sim_f = 1000):
    encoders = []
    for _ in range(n_enc):
        e = Encoder(min_f, max_f, sim_f)
        encoders.append(e)

    return encoders

def run_network(neurons, encoders, enc_input, sim_time):

    for i, e in enumerate(encoders):
        e.set_value(enc_input[i])

    # create a 2D array which represents the spike raster
    # where the first dimension is which neuron is spiking
    # and the second dimension is a list of all the times it spiked
    neuron_fires = []
    encoder_fires = []
    for i in range(len(neurons)):
        neuron_fires.append([])

    for i in range(len(encoders)):
        encoder_fires.append([])
    
    # simulate
    for t in range(sim_time):
        # if an encoder spikes, call spike on its neuron
        for i, e in enumerate(encoders):
            if e.update():
                encoder_fires[i].append(t)
                neurons[i].spike()

        # update the network
        for i, n in enumerate(neurons):
            if n.update():
                neuron_fires[i].append(t)

    return neuron_fires, encoder_fires

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
            if np.isclose(m1, m2) and m1 != 0:
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
            if np.isclose(m1, m2) and m1 != 0:
                break

    return np.asarray(fire_matrix)

def save_trained_network(filename, neurons, encoders, dpe_weights, window_size, sim_time, c_acc, E_t, avg_ss):
    network = {}

    network['neurons'] = []
    network['synapses'] = []
    network['encoders'] = []
    network['dpe_weights'] = list(dpe_weights.flatten())
    network['window_size'] = window_size
    network['sim_time'] = sim_time
    network['cumulative accuracy'] = c_acc
    network['Epoch Accuracy'] = E_t
    network['Average Steady State Time'] = avg_ss

    for n in neurons:
        saved_n = {}
        saved_n['leak'] = n.leak
        saved_n['threshold'] = n.threshold
        network['neurons'].append(saved_n)

        for s in n.synapses:
            saved_s = {}
            saved_s['n1'] = s.n1.id
            saved_s['n2'] = s.n2.id
            saved_s['weight'] = s.weight

            network['synapses'].append(saved_s)

    if encoders is not None:
        for i, e in enumerate(encoders):
            saved_e = {}
            saved_e['min_f'] = e.min_f
            saved_e['max_f'] = e.max_f
            saved_e['sim_f'] = e.sim_f
            saved_e['enc_type'] = e.enc_type
            network['encoders'].append(saved_e)

    with open(filename, 'w') as f:
        f.write(json.dumps(network))

def load_trained_network(filename):
    network = {}
    with open(filename, 'r') as f:
        network = json.load(f)

    neurons = []
    encoders = []
    dpe_weights = np.asarray(network['dpe_weights']).reshape((len(network['neurons']), -1))
    window_size = network['window_size']
    sim_time = network['sim_time']
    c_acc = network['cumulative accuracy']
    E_t = network['Epoch Accuracy']
    avg_ss = network['Average Steady State Time']

    for i, n in enumerate(network['neurons']):
        loaded_n = Neuron(i, n['threshold'], n['leak'])
        neurons.append(loaded_n)

    for s in network['synapses']:
        n1 = neurons[s['n1']]
        n2 = neurons[s['n2']]
        loaded_s = Synapse(n1, n2, s['weight'])

        neurons[s['n1']].add_synapse(loaded_s)

    for e in network['encoders']:
        loaded_e = Encoder(e['min_f'], e['max_f'], e['sim_f'])
        encoders.append(loaded_e)

    
    return neurons, encoders, dpe_weights, window_size, sim_time, c_acc, E_t, avg_ss

# simulates a network with timeseries data, at each timestep the first n_input neurons receive potentiation in the form of the data
# NOTE: the input neurons are divided evenly into two types. The first get the data normally and the second get an inverted version of the data. 
#   This is done so that the spike raster is not empty in the sections where input is low
# returns - spike raster where the rows correspond to a neuron and each column is a timestep. 
#   If the neuron fired at that timestep there is a 1 otherwise there is a 0
def run_network_timeseries(neurons, data, n_input):
    # simulate
    spike_raster = []
    for i in range(len(data)):
        spike_raster.append([])

    # feed a peice of data in at each timestep
    for t in range(len(data)):
        # get the input for this timestep, and apply it to input neurons
        for i in range(int(n_input/2)): #normal
            neurons[i].apply_potential(data[t])
        for i in range(int(n_input/2)): #inverted
            neurons[i+int(n_input/2)].apply_potential(-data[t]+1)

        # update the network
        for n in neurons:
            if n.update():
                spike_raster[t].append(1)
            else:
                spike_raster[t].append(0)

    return np.asarray(spike_raster)

# same as run_network_timeseries but for n-dimensional input
def run_network_timeseries_nD(neurons, data, n_input):
    # simulate
    spike_raster = []
    for i in range(len(data)):
        spike_raster.append([])

    # number of input neurons per dim (eg. 3 dim, 12 inp -> 2)
    n_per_dim = int(n_input/2/len(data[0]))

    # feed a peice of data in at each timestep
    for t in range(len(data)):

        # get the input for this timestep, and apply it to input neurons
        #   feed each dim in
        for i, d in enumerate(data[t]):
            # normally to n_per_dim neurons
            for j in range(n_per_dim):
                neurons[j + i*n_per_dim].apply_potential(d) #normal

            # inverted to n_per_dim neurons offset by half of n_input
            for j in range(n_per_dim):
                neurons[(j + i*n_per_dim) + int(n_input/2)].apply_potential(-d) #inverted

        # update the network
        for n in neurons:
            if n.update():
                spike_raster[t].append(1)
            else:
                spike_raster[t].append(0)

    return np.asarray(spike_raster)

