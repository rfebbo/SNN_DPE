import numpy as np
from tqdm import tqdm

from snn_dpe.tools.network import (reset_network, run_network_early_exit,
                                   run_network_timeseries)

import matplotlib.pyplot as plt

def mse(y, y_hat):
    return np.sum((y - y_hat)**2)

# find the average spiking rate of each neuron (x) and multiply it with the dpe weights
def forward_pass(fire_matrix, dpe_weights, bias=None):
    x = np.mean(fire_matrix, axis=0)
    y = np.dot(x, dpe_weights)
    # y = np.sum(y, axis=0)

    if bias:
        return x, y + bias
    else:
        return x, y

def update_weights(dpe_weights, x, y, y_hat, lr=0.005, bias=None):
    n_classes = dpe_weights.shape[1]

    e = 2 * (y - y_hat)

    if bias:
        bias -= np.sum(e) * lr

    for i in range(len(x)):
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
        for l, d in zip(labels, data):
            # run network
            fire_matrix = run_network_early_exit(neurons, encoders, d, sim_time, window_size=window_size)

            reset_network(neurons, encoders)

            steady_state_t = len(fire_matrix)

            x, y = forward_pass(fire_matrix, dpe_weights)

            y_hat = np.zeros(len(classes))
            y_hat[l] = 1

            update_weights(dpe_weights, x, y, y_hat)
            
            correct = 0
            if np.argmax(y) == l:
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

# uses io pairs and an SNN to train a dpe layer
def train_TS(n_epochs, TS_inputs, TS_outputs, neurons, n_input, write_noise_std = 0, silent=True, TS_inputs_te = None, TS_outputs_te = None, initial_weights = None, initial_bias = None, relative=False, post_sample_reset=True, reset_synapses=True, post_epoch_reset=False):
    if initial_weights == None or initial_bias == None:
        dpe_weights = np.random.rand(len(neurons), len(TS_outputs[0]))
        dpe_bias = np.random.rand(len(TS_outputs[0]))

        dpe_weights = np.asarray(dpe_weights, dtype=np.float16)
    else:
        dpe_weights = initial_weights
        dpe_bias = initial_bias

    # track mean squared errors
    Training_mses = []
    Testing_mses = []

    
    
    for _ in range(n_epochs):
        # progress bar for each epoch
        pbar = tqdm(TS_inputs, disable=silent)

        mse_avg = 0
        n_processed = 0
        if len(Testing_mses) == 0:
            Testing_mses.append('N/A')

        # iterate over the data
        for input_i, output_i in zip(pbar, TS_outputs):

            # run the network and get the spike raster
            spike_raster = run_network_timeseries(neurons, input_i, n_input)
            
            if post_sample_reset:
                for n in neurons:
                    n.reset(reset_synapses)

            # use the spike raster as an input for the dpe and calculate the result
            x, y = forward_pass(spike_raster, dpe_weights, bias=dpe_bias)

            # relative mode will train the dpe to find the delta from the 
            # last timestep instead of the absolute value
            if relative:
                desired_output = output_i - input_i[-1]
            else:
                desired_output = output_i

            # use gradient descent to update the weights
            update_weights(dpe_weights, x, y, desired_output, bias=dpe_bias)

            dpe_weights = np.random.normal(dpe_weights, write_noise_std, size=dpe_weights.shape)            
            dpe_bias = np.random.normal(dpe_bias, write_noise_std, size=dpe_bias.shape)            
            # dpe_weights += np.random.choice([-1,1], dpe_weights.shape)*dpe_weights*write_noise_std
            # dpe_bias += np.random.choice([-1,1], dpe_bias.shape)*dpe_bias*write_noise_std

            # calculate cumulative average mse
            if mse_avg == 0:
                mse_avg = mse(y, desired_output)
            else:
                mse_avg = (mse(y, desired_output) + n_processed * mse_avg)/ (n_processed+1)

            n_processed += 1
            pbar.set_description(f"Training MSE: {mse_avg:.4}, Testing MSE: {Testing_mses[-1]:.4}")


        if post_epoch_reset:
            for n in neurons:
                n.reset(reset_synapses)

        Training_mses.append(mse_avg)
        if TS_inputs_te is not None:
            Testing_mse = test_timeseries(TS_inputs_te, TS_outputs_te, neurons, dpe_weights, n_input, bias=dpe_bias, plot_len=0, relative=relative)
            if Testing_mses[-1] == 'N/A':
                Testing_mses[-1] = Testing_mse
            else:
                Testing_mses.append(Testing_mse)

    return Training_mses, Testing_mses, dpe_weights, dpe_bias



def predict(neurons, encoders, dpe_weights, sample, sim_time=100, window_size=10):
    # run network
    fire_matrix = run_network_early_exit(neurons, encoders, sample, sim_time, window_size=window_size)

    reset_network(neurons, encoders)

    _, y = forward_pass(fire_matrix, dpe_weights)

    return y


# takes inputs of specific window size and calculates outputs for testing then plots if plt_len > 0
def test_timeseries(TS_inputs, TS_outputs, neurons, dpe_weights, n_input, stride=1, TS_data = None, bias = None, relative=False, plot_len = 200, post_sample_reset=True, reset_synapses=True):
    
    warmup = len(TS_inputs[0])
    testing_mse = 0

    # for plotting
    dpe_outputs = []
    dpe_time = []
    
    for i, (input_sample, output_sample) in enumerate(zip(TS_inputs, TS_outputs)):
        t = i*stride+warmup

        spike_raster = run_network_timeseries(neurons, input_sample, n_input)

        if post_sample_reset:
            for n in neurons:
                n.reset(reset_synapses)

        _, y = forward_pass(spike_raster, dpe_weights, bias=bias)


        if relative:
            dpe_output = input_sample[-1] + y
        else:
            dpe_output = y

        testing_mse += mse(dpe_output, output_sample)

        if t < plot_len:        
            dpe_time.append(t)
            dpe_outputs.append(dpe_output)


    if plot_len > 0 and TS_data is not None:
        plt.plot(range(len(TS_data))[:plot_len], TS_data[:plot_len])
        plt.plot(dpe_time, dpe_outputs)
        plt.show()

    return testing_mse/len(TS_inputs)