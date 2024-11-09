import neuro
import risp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def plot_network(net):
    # visualize
    G = nx.Graph()

    for n in net.nodes():
        G.add_node(n)

    for e in net.edges():
        G.add_edge(e[0], e[1])

    nx.draw(G)
    # plt.savefig('Images/network.pdf')
    plt.show()

def create_network():
    proc_params = {
        "leak_mode": "none",
        "min_weight": -1,
        "max_weight": 1,
        "min_threshold": -1,
        "max_threshold": 1,
        "max_delay": 5,
        "discrete": False,
        "min_potential": -1
    }

    proc = risp.Processor(proc_params)

    np.random.seed()
    seed = np.random.randint(0,10000000000)
    n_neurons = 50
    connectivity = 0.05
    n_synapses = int(connectivity * (n_neurons**2))

    w_plus = 0.5
    w_minus = 0.25
    vt = 1.0
    delay = 1

    net = neuro.Network()
    net.set_properties(proc.get_network_properties())
    moa = neuro.MOA()
    moa.seed(seed, "reservoir")

    # print(dir(net))
    for i in range(n_neurons):
        node = net.add_node(i)
        node.values[0] = vt

    while len(list(net.edges())) < n_synapses:
        e = np.random.choice(n_neurons, 2)
        
        try:
            edge = net.add_edge(e[0], e[1])
            edge.values[0] = np.random.choice([w_plus, w_minus], 1)
            edge.values[1] = delay
        except:
            continue

        
    # plot_network(net)
    return net

net = create_network()