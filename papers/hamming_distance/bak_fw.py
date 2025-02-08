import ravens

def create_ravens_network(\
        n_input : int, n_hidden : int, n_output : int,\
        connectivity : float, w_plus : float, w_minus : float):

    proc_params = {
        "min_weight": -8,
        "max_weight": 7,

        "max_delay": 8,

        "min_threshold": 0,
        "max_threshold": 15,

        "min_standard_resting_potential": 0,
        "max_standard_resting_potential": 0,

        "min_refractory_resting_potential": -5,
        "max_refractory_resting_potential": 0,

        "min_absolute_refractory_period": 0,
        "max_absolute_refractory_period": 4,

        "min_relative_refractory_period": 0,
        "max_relative_refractory_period": 5,

        "min_leak": 0,
        "max_leak": 2,
        "stdp": [1, 2, 2, 3, 4, -4, -2, -1],
        "spike_value_factor": 16
    }


    proc = ravens.Processor(proc_params)

    np.random.seed()
    seed = np.random.randint(0,1000000000)
    n_neurons = n_input + n_hidden + n_output
    n_synapses = int(connectivity * (n_neurons**2))

    vt = 4
    delay = 1

    net = neuro.Network()
    net.set_properties(proc.get_network_properties())
    moa = neuro.MOA()
    moa.seed(seed, "reservoir")

    for i in range(n_input):
        node = net.add_node(i)
        node.values[0] = vt
        node.values[1] = 0
        node.values[2] = 0
        node.values[3] = 0
        node.values[4] = 0
        node.values[5] = 0
        net.add_input(i)

    for i in range(n_input, n_hidden):
        node = net.add_node(i)
        node.values[0] = vt

    for i in range(n_input + n_hidden, n_output):
        node = net.add_node(i)
        node.values[0] = vt
        net.add_output(i)
        

    while len(list(net.edges())) < n_synapses:
        e = np.random.choice(n_neurons, 2)
        
        try:
            edge = net.add_edge(e[0], e[1])
            edge.values[0] = np.random.choice([w_plus, w_minus], 1)
            edge.values[1] = delay
        except:
            continue

        
    proc.load_network(net)

    for i in range(n_neurons):
        proc.track_neuron_events(i)

    for i in range(n_output):
        print(proc.track_output_events(i))

    return net, proc

