import neuro
import risp
import eons
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

def net_char(net):

    N = {}
    for n in net.nodes():
        N[n] = 0

    for e in net.edges():
        N[e[1]] += 1

    n_inputs = 0
    for n in N:
        n_inputs += N[n]

    print(n_inputs/len(list(net.nodes())))


def create_risp_network(\
        n_input : int, n_hidden : int, n_output : int,\
        n_synapses : int, vt : float, w_plus : float, w_minus : float):

    proc_params = {
        "leak_mode": "none",
        "min_weight": w_minus,
        "max_weight": w_plus,
        "min_threshold": 0,
        "max_threshold": vt,
        "max_delay": 1,
        "discrete": False,
        "min_potential": -1, 
        "spike_value_factor" : 1
    }


    # proc_params = {
    #     "leak_mode": "none",
    #     "min_weight": -1,
    #     "max_weight": 1,
    #     "min_threshold": -1,
    #     "max_threshold": 1,
    #     "max_delay": 5,
    #     "discrete": False,
    #     "min_potential": -1
    # }
    proc = risp.Processor(proc_params)

    np.random.seed()
    seed = np.random.randint(0,1000000000)
    n_neurons = n_input + n_hidden + n_output

    delay = 1

    net = neuro.Network()
    net.set_properties(proc.get_network_properties())
    moa = neuro.MOA()
    moa.seed(seed, "reservoir")

    for i in range(n_input):
        node = net.add_node(i)
        node.values[0] = vt
        net.add_input(i)

    for i in range(n_input, n_input + n_output):
        node = net.add_node(i)
        node.values[0] = vt
        net.add_output(i)

    for i in range(n_input + n_output, n_input + n_output + n_hidden):
        node = net.add_node(i)
        node.values[0] = vt
        

    while len(list(net.edges())) < n_synapses:
        e1 = np.random.choice(n_neurons, 1)
        e2 = np.random.choice(np.arange(n_input, n_neurons), 1)
        
        try:
            edge = net.add_edge(e1, e2)
            edge.values[0] = np.random.choice([w_plus, w_minus], 1)
            edge.values[1] = delay
        except:
            continue

    net.as_json()
    proc.load_network(net)

    neuro.track_all_neuron_events(proc, net)
    neuro.track_all_output_events(proc, net)
    

    return net, proc


n_neurons = 100
p_input = 0.1
p_output = 0.1
n_input = int(n_neurons * p_input)
n_output = int(n_neurons * p_output)
n_hidden = n_neurons - n_input - n_output

# A = average number of inputs per neuron
# B = average # of inputs required to spike
# gamma = A/B 
#   gamma > 1 sensitive network
#   gamma < 1 insensitive network


vt = 32
w_plus = vt / 4
w_minus = -(w_plus * 0.5)

B = vt / ((w_plus + w_minus) / 2)

n_synapses = B * n_neurons
A = n_synapses / n_neurons

if n_synapses > n_neurons ** 2:
    raise Exception("not enough neurons")

print(A)
print(B)

gamma = A / B
print(gamma)

net, proc = create_risp_network(n_input, n_hidden, n_output, n_synapses, vt, w_plus, w_minus)

# net_char(net) 
for i in range(n_input):
    for j in range(vt):
        s = neuro.Spike(i, j, 1.0)
        proc.apply_spike(s)

proc.run(vt + 20)

print(proc.neuron_charges())

sr = proc.neuron_vectors()


for i, n in enumerate(sr):
    plt.scatter(n, np.ones(len(n)) * i, marker='|')

plt.show()

plot_network(net)