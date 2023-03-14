import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from snn_dpe.tools.network import run_network_timeseries
from snn_dpe.tools.train.utils import forward_pass, mse, rmse, update_weights


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