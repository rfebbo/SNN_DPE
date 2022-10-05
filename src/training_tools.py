import numpy as np
from network_tools import reset_network, run_network_early_exit

def mse(y, y_hat):
    return np.sum((y - y_hat)**2)

# find the average spiking rate of each neuron (x) and multiply it with the dpe weights
def forward_pass(fire_matrix, dpe_weights):
    x = np.mean(fire_matrix, axis=0)
    y = np.dot(x, dpe_weights)
    return x, y

def update_weights(fire_matrix, dpe_weights, x, y, y_hat, lr=0.005):
    n_neurons = len(fire_matrix[0])
    n_classes = dpe_weights.shape[1]

    e = 2 * (y - y_hat)

    for i in range(n_neurons):
        for j in range(n_classes):
            dpe_weights[i][j] -= e[j] * x[i] * lr

def train_all(data, labels, classes, neurons, encoders, dpe_weights, sim_time = 200, window_size=10, n_epochs=10):
    E_t = []
    cumulative_acc = []
    avg_steady_state_t = 0
    n_proccessed = 0
    
    for _ in range(n_epochs):
        # for sample in data
        n_correct = 0
        for i, d in enumerate(data):
            # run network
            fire_matrix = run_network_early_exit(neurons, encoders, d, sim_time, window_size=window_size)

            reset_network(neurons, encoders)

            steady_state_t = len(fire_matrix)

            x, y = forward_pass(fire_matrix, dpe_weights)

            y_hat = np.zeros(len(classes))
            y_hat[labels[i]] = 1

            update_weights(fire_matrix, dpe_weights, x, y, y_hat)
            
            correct = 0
            if np.argmax(y) == labels[i]:
                n_correct += 1
                correct = 1

            if n_proccessed == 0:
                avg_steady_state_t = steady_state_t
                cumulative_acc.append(correct)    
            else:
                avg_steady_state_t = (n_proccessed * avg_steady_state_t + steady_state_t)/(n_proccessed+1)
                cumulative_avg = (n_proccessed * cumulative_acc[-1] + float(correct))/(n_proccessed + 1)
                cumulative_acc.append(cumulative_avg)

            n_proccessed += 1
                
        E_t.append(float(n_correct)/len(data))

    return E_t, avg_steady_state_t, cumulative_acc