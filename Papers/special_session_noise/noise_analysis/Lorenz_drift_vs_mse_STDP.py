# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from snn_dpe.tools.data import util, Lorenz
from snn_dpe.tools.network import create_network, run_network_timeseries_nD, reset_network
from snn_dpe.tools.plotting import *
from snn_dpe.tools.plotting import plot_network, plot_spike_raster
from snn_dpe.tools.train.timeseries import (test_timeseries_nD,
                                            train_TS_nD)

# slice up MG into input, output pairs
input_window_size = 10
output_window_size = 1

n_tr_data = 2000
n_te_data = 1000


# NOTE: tmax/n defines the resolution of the data and the speed of the particle
LD_data = Lorenz.create_dataset(x0=10,y0=11,z0=11.05,tmax=50,n=n_tr_data+n_te_data, normalize=False)
LD_max = np.max(LD_data)
LD_min = np.min(LD_data)
# split data into training and testing
LD_data_tr = LD_data[:n_tr_data] 
LD_data_te = LD_data[n_te_data:] 

LD_inputs_tr, LD_outputs_tr = util.generate_io_pairs(LD_data_tr, input_window_size, output_window_size)
LD_inputs_te, LD_outputs_te = util.generate_io_pairs(LD_data_te, input_window_size, output_window_size)


def run_test(args):
    tr_mses = []
    te_mses = []

    np.random.seed()

    noise_axis = np.linspace(0, 0.1, num=40)
    drift_axis = np.linspace(0, 0.02, num=40)
    n=0
    #for n in noise_axis:
    for d in drift_axis:

        # create a test network
        n_neurons = 30
        synapse_density = 0.1
        n_input = 12 #how many neurons will receive potentiation adjustments related to MG
        n_synapses = int((n_neurons**2)*synapse_density)

        # dpe noise
        write_noise = n
        # synapse parameters
        drift = d
        synapse_noise = 0
        stdp=True

        # training parameters
        n_epochs = 5
        post_sample_reset = True #also applies to testing
        reset_synapses = False #also applies to testing
        post_epoch_reset = False
        LD_inputs_te_noise = LD_inputs_te + np.random.normal(0, 0) * (LD_max-LD_min)

        neurons = create_network(n_neurons, n_synapses, negative_weights = True, threshold_range = (0.35, 1), leak_range = (0.05, 0.25), weight_factor = 1, std_dev=synapse_noise, drift = drift, stdp=stdp)
        # train
        training_mses, testing_mses, _, _ = train_TS_nD(n_epochs, LD_inputs_tr, LD_outputs_tr, neurons, n_input, silent=True, TS_inputs_te=LD_inputs_te_noise, TS_outputs_te=LD_outputs_te, relative=True, write_noise_std=write_noise, post_sample_reset=post_sample_reset, reset_synapses=reset_synapses, post_epoch_reset=post_epoch_reset)
        
        # save the final mses of the last epoch
        tr_mses.append(training_mses[-1] / (LD_max-LD_min))
        te_mses.append(testing_mses[-1] / (LD_max-LD_min))

    return tr_mses, te_mses, drift_axis

if __name__ == '__main__':

    n_tests = 60
    n_threads = 16
    with Pool(processes=n_threads) as p:
        results = list(tqdm(p.imap(run_test, range(n_tests)), total=n_tests))
        

    import csv

    for i, r in enumerate(results):
        with open(f'./noise_results/Lorenz/drift/LD_drift_vs_MSE_STDP_{i}.csv', 'w') as f:
            wtr = csv.writer(f, delimiter=',', lineterminator='\n')
            
            for data in zip(r[0], r[1], r[2]):
                wtr.writerow(list(data))
