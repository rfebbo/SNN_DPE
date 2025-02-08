from pdb import run
from snn_dpe.neuron_synapse import Neuron, Synapse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from snn_dpe.tools.data import normalize_iris_data, read_iris_data
from snn_dpe.tools.network import create_encoders

def plot_network(neruons, synapses):
    # visualize
    G = nx.Graph()

    pos = []
    for row in neruons:
        for n in row:
            G.add_node(n['id'])
            pos.append(n['pos'])

    # for e in net.edges():
    #     G.add_edge(e[0], e[1])

    for s in synapses:
        G.add_edge(s.n1.id, s.n2.id)

    fig, ax = plt.subplots(1,1, figsize=(40,4), dpi=80)
    nx.draw(G, pos, node_size=0.5, width = 0.2)
    # plt.savefig('Images/network.pdf')
    plt.show()

def create_2d_snn(width, height):
    neurons = []
    network = []

    id = 0
    for wi in range(width):
        neurons.append([])
        for hi in range(height):
            n = Neuron(threshold=0.75, leak=0.01)
            n.id = id
            id += 1
            pos = [wi, hi]
            network.append(n)
            neurons[wi].append({'neuron' : n, 'pos' : pos, 'id' : n.id})
            
    max_distance = 10
    n_neurons = width * height
    p_output = 0.1
    n_input = height
    n_output = int(n_neurons * p_output)
    n_hidden = n_neurons - n_input - n_output

    vt = 1
    w_plus = 0.5
    w_minus = -0.05

    B = vt / ((w_plus + w_minus) / 2)

    n_synapses = B * n_neurons
    A = n_synapses / n_neurons
    n_synapses = int(n_synapses)

    
    horizontal_mean = 10
    horizontal_std = 5

    vertical_mean = 0

    vertical_std = 5
    
    n_samples = 40

    x_distances = np.random.normal(horizontal_mean, horizontal_std, n_synapses*n_samples).astype(np.int8)
    y_distances = np.random.normal(vertical_mean, vertical_std, n_synapses*n_samples).astype(np.int8)


    im = []
    for y in np.unique(y_distances):
        im.append([])
        for x in np.unique(x_distances):
            count = np.sum(np.bitwise_and(x_distances == x, y_distances == y))
            im[-1].append(count)

    plt.imshow(im)
    print(f'desired im: {np.sum(im)}')
    print(f'desired distr: {len(x_distances)}')
    plt.yticks(range(len(np.unique(y_distances))), np.unique(y_distances))
    plt.xticks(range(len(np.unique(x_distances))), np.unique(x_distances))
    plt.tick_params(axis='both', labelsize=5)
    plt.xlabel('Horizontal Distance from neuron')
    plt.ylabel('Vertical Distance from neuron')
    plt.title(f'Desired Synapse Distance Distribution \n target: ({horizontal_mean},{vertical_mean})')
    plt.colorbar()
    plt.show()
    print(n_synapses)
    print(n_synapses/(n_neurons**2))

    # plt.plot(distance)
    # plt.show()
    synapses = []
    syn_distances_x = []
    syn_distances_y = []
    di = 0
    while len(synapses) < n_synapses:
        # select a random neuron
        pre_n_x = np.random.randint(0, width)
        pre_n_y = np.random.randint(0, height)

        pre_n = neurons[pre_n_x][pre_n_y]['neuron']

        post_n_x = pre_n_x + x_distances[di]
        post_n_y = pre_n_y + y_distances[di]


        if post_n_x < 0:
            di += 1
            di = di % n_synapses
            continue
        if post_n_x >= width:
            di += 1
            di = di % n_synapses
            continue
        if post_n_y < 0:
            di += 1
            di = di % n_synapses
            continue
        if post_n_y >= height:
            di += 1
            di = di % n_synapses
            continue

        post_n = neurons[post_n_x][post_n_y]['neuron']
        syn_distances_x.append(x_distances[di])
        syn_distances_y.append(y_distances[di])
        
        s = Synapse(pre_n, post_n, weight=np.random.choice([w_plus, w_minus]), delay=1)
        pre_n.add_synapse(s)
        synapses.append(s)
        di += 1
        di = di % n_synapses
        
    
    im2 = []
    for synd_y in np.unique(syn_distances_y):
        im2.append([])
        for synd_x in np.unique(syn_distances_x):
            count = np.sum(np.bitwise_and(syn_distances_x == synd_x,  syn_distances_y == synd_y))
            im2[-1].append(count)
            
    plt.imshow(im2)
    print(np.sum(im2))
    plt.yticks(range(len(np.unique(syn_distances_y))), -np.unique(syn_distances_y))
    plt.xticks(range(len(np.unique(syn_distances_x))), np.unique(syn_distances_x))
    plt.tick_params(axis='both', labelsize=5)
    plt.xlabel('Horizontal Distance from neuron')
    plt.ylabel('Vertical Distance from neuron')
    plt.title(f'Actual Synapse Distance Distribution \n target: ({horizontal_mean},{vertical_mean})')
    plt.colorbar()
    plt.show()
        
    return neurons, synapses, network

def plot_spikes(neuron_fires, encoder_fires, attributes, einputs, width):
    fig, ax = plt.subplots(2, 1, figsize=(15,5), sharex=True, gridspec_kw={'height_ratios' : [1, 10]})
    
    for i, f in enumerate(encoder_fires):
        # plot encoders on separate axis
        ax[0].scatter(f, np.ones(len(f)) * i, marker='|')

    for i, f in enumerate(neuron_fires):
        ax[1].scatter(f, np.ones(len(f)) * (i), marker='|')


    enc_labels = [f'neuron {i}' for i in range(len(encoder_fires))]

    font_size = 7
    ax[0].set_ylim(-0.5, len(attributes) - 0.5)
    ax[0].set_yticks(ticks=range(len(attributes)), labels=enc_labels)
    ax[0].tick_params(axis='both', labelsize=font_size)
    ax[0].set_title('encoder spikes', fontsize=font_size)

    y_ticks = np.linspace(0, len(neuron_fires), 10, dtype=np.int16)
    y_ticks_labels = np.linspace(0, width, 10, dtype=np.int16)
    ax[1].set_ylim(-0.5, len(neuron_fires) - 0.5)
    ax[1].set_yticks(ticks=y_ticks, labels=y_ticks_labels)
    ax[1].tick_params(axis='both', labelsize=font_size)
    ax[1].set_ylabel('Distance from Input', fontsize=font_size)
    ax[1].set_title('neuron spikes', fontsize=font_size)


    ax[1].set_xlabel('Time (ms)', fontsize=font_size)
    # plt.tight_layout()
    # plt.savefig('Images/Network_output_w_enc.pdf')
    plt.show()



def run_2d_snn(network, width):
    n_neruons = len(network)
    # load iris data
    iris_data_location = '../../Datasets/Iris/iris.data'

    iris_data, labels, classes, attributes = read_iris_data(iris_data_location, shuffle=False)

    normalized_iris_data = normalize_iris_data(iris_data, attributes)

    min_f = 100
    max_f = 800
    sim_f = 1000

    sim_time = 200

    encoders = create_encoders(len(attributes), min_f, max_f, sim_f)

    enc_inputs = normalized_iris_data[0]



    for i, e in enumerate(encoders):
            e.set_value(enc_inputs[i])

    # create a 2D array which represents the spike raster
    # where the first dimension is which neuron is spiking
    # and the second dimension is a list of all the times it spiked
    neuron_fires = []
    encoder_fires = []
    for i in range(n_neurons):
        neuron_fires.append([])

    for i in range(len(encoders)):
        encoder_fires.append([])

    # simulate
    data_sample = 0
    for t in range(sim_time):
        if t % 50 == 0 and t != 0:
            data_sample += 1
            print(data_sample)
            enc_inputs = normalized_iris_data[data_sample % len(normalized_iris_data)]
            for i, e in enumerate(encoders):
                e.reset()
                e.set_value(enc_inputs[i])
            
            for i, n in enumerate(network):
                n.reset()
            

        # if an encoder spikes, call spike on its neuron
        for i, e in enumerate(encoders):
            if e.update():
                encoder_fires[i].append(t)
                # network[i].spike()
                network[i].apply_potential(network[i].threshold)

        # update the network
        for i, n in enumerate(network):
            if n.update():
                neuron_fires[i].append(t)



    # reset_network(neurons, encoders)

    plot_spikes(neuron_fires, encoder_fires, attributes, enc_inputs, width)

# 

width = 50
height = 10
n_neurons = width * height

neurons, synapses, network = create_2d_snn(width, height)
    
# plot_network(neurons, synapses)

# run_2d_snn(network, width)