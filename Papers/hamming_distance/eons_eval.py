import neuro
import eons

# Set up the initial EONS parameters
eons_params = {
    "merge_rate": 0.1,
    "population_size": 1000,
    "multi_edges": 0,
    "crossover_rate": 0.5,
    "mutation_rate": 0.9,
    "starting_nodes": 20,
    "starting_edges": 25,
    "selection_type": "tournament",
    "tournament_size_factor": 0.1,
    "tournament_best_net_factor": 0.9,
    "random_factor": 0.5,
    "num_best": 5,
    "num_mutations": 1,
    "node_mutations": {}, # No node parameters
    "net_mutations": {},  # No network parameters
    "edge_mutations": { }, # No edge parameters
    "add_node_rate" : 100, 
    "add_edge_rate" : 100,
    "delete_node_rate" : 10,
    "delete_edge_rate" : 10,
    "net_params_rate" : 0,
    "node_params_rate" : 0,
    "edge_params_rate" : 0,
    "seed_eo":0
}

template_genome = neuro.Network()

template_genome.add_node(0)
template_genome.add_input(0)

template_genome.add_node(1)
template_genome.add_input(1)

template_genome.add_node(2)
template_genome.add_output(2)

evolver = eons.EONS(eons_params)
evolver.set_template_network(template_genome)

# order: the order in which the nodes should be visited
# out: A dictionary mapping each node ID to all of its outputs
# v: the vector of input values to start with

def calculate_nand_tree(order, out, v):
    received_inputs = {}
    for i in out.keys():
        received_inputs[i] = []
        
    received_inputs[0] = [v[0]]
    received_inputs[1] = [v[1]]
    
    for i in order:
        # If it doesn't receive any input at all, it will produce 0
        if (len(received_inputs[i]) == 0):
            output = 0
            
        # If it only receives one input, if it's an input neuron, it will pass that value directly
        # Otherwise, it treats it as if the value was passed into the NAND gate twice, 
        # so it performs a NOT operation
        elif (len(received_inputs[i]) == 1):
            if (i == 1 or i == 0):
                output = received_inputs[i][0]
            else:
                output = not received_inputs[i][0]
        # If it receives two or more inputs, it will perform the NAND operation across all inputs
        else:          
            output =  not received_inputs[i][0] or not received_inputs[i][1]
            for j in range(2,len(received_inputs[i])):
                output = output or not received_inputs[i][j]
                
        # Converts the True/False values to 1s and 0s for the rest of the calculations
        output = int(output)
        for j in out[i]:
            received_inputs[j].append(output)
            
    return output

def fitness(g):
    
    # The function we're trying to replicate.  Here it is XOR, but you can update the correct output 
    # to reflect the behavior of other logic gates.
    values = [[0,0], [0,1], [1,0], [1,1]]
    correct = [0, 1, 1, 0]
    
    # Remove loops (self-edges) in the graph
    for node in g.nodes():
        if (g.is_edge(node,node)):
            g.remove_edge(node,node)
    
    # Remove outgoing edges from output
    node = g.get_node(2)
    edges = node.outgoing
    for i in range(len(edges)):
        e = edges[i]
        g.remove_edge(e.pre.id,e.post.id)

    # Remove incoming edges from input 
    for i in range(0,2):
        node = g.get_node(i)
        edges = node.incoming
        for j in range(len(edges)):
            e = edges[j]
            g.remove_edge(e.pre.id,e.post.id)
            
    g.prune()
    
    # Modified BFS to only visit a node when ALL of its incoming nodes have already been visited
    stack = [0, 1]
    visited = []
    while (len(stack) != 0):
        id = stack[0]
        visited.append(id)
        stack = stack[1:]
        node = g.get_node(id)
        for i in range(len(node.outgoing)):
            add_it = True
            e = node.outgoing[i]
            next_id = e.post.id
            if (next_id not in visited and next_id not in stack):
                nn = g.get_node(next_id)
                for j in range(len(nn.incoming)):
                    
                    pre_id = nn.incoming[j].pre.id
                    if (pre_id not in visited):
                        add_it = False
                if (add_it == True):
                    stack.append(next_id)
    
    # Can't get to the output from inputs, which is bad!
    if (2 not in visited):
        return -100
    
#     visited.remove(2)
#     visited.append(2)
    
    # Construct vectors of inputs and outputs for each node
    outputs = {}
    inputs = {}
    for node in g.nodes():
        outputs[node] = []
        n = g.get_node(node)
        outputs[n.id] = []
        inputs[n.id] = []
        for i in range(len(n.outgoing)):
            e = n.outgoing[i]
            outputs[n.id].append(n.outgoing[i].post.id)
        for i in range(len(n.incoming)):
            e = n.incoming[i]
            inputs[n.id].append(n.incoming[i].pre.id)

    # Check to make sure that at least one of the gates is receiving more than one input and 
    # thus functioning as a NAND gate.  If not, return a low score.
    count = 0
    for n in inputs.keys():
        if (len(inputs[n]) >= 2):
            count += 1
            break

    if (count == 0):
        return -10
    
    # If either of the input values are not being used, return a low score. 
    if (len(outputs[0]) == 0 or len(outputs[1]) == 0):
        return -100
    
    # Now we can calculate the output value of the graph of NAND gates.
    fit = 0
    
    for i in range(len(values)):
        v = values[i]
        output = calculate_nand_tree(visited, outputs, v)
        if (output == correct[i]):
            fit += 1
    
    # If all of the fitness values are correct, we set the score to be 100 and then we set EONS to minimize
    # the size of the network by adding the inverse of the number of nodes in the graph to the fitness score.
    if (fit == len(values)):
        fit = 100+1.0/g.num_nodes()
        return fit
    
#     if (count > 1):
#         fit += 0.5

        
    return fit

pop = evolver.generate_population(eons_params, 42)

print(dir(pop.networks[0]))
print(pop.networks[0].network)

# vals = []
# for i in range(50):
#     # Calculate the fitnesses of all of the networks in the population
#     fitnesses = [fitness(g.network) for g in pop.networks]
    
#     # Get information about the best network
#     max_fit = max(fitnesses)
#     vals.append(max_fit)
#     index = fitnesses.index(max_fit)
#     gmax = pop.networks[index].network
    
#     # Switch the priorities of EONS to minimize network size once the desired behavior has been reached
#     if (max_fit > 100 and eons_params["merge_rate"] != 0):
#         eons_params["merge_rate"] = 0
#         eons_params["crossover_rate"] = 0
#         eons_params["add_node_rate"] = 10
#         eons_params["add_edge_rate"] = 10
#         eons_params["delete_node_rate"] = 100
#         eons_params["delete_edge_rate"] = 100
    
#     # Print statistics about the performance
#     print("==========================Epoch {:4d}: {} {}".format(i, max_fit, gmax.num_nodes()))

    
#     # Create the next population based on the fitnesses of the current population
#     pop = evolver.do_epoch(pop, fitnesses, eons_params)
    
# if (max_fit > 100):
#     print(gmax)