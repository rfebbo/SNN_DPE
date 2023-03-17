from multiprocessing.pool import Pool
import csv


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from snn_dpe.tools.data import Mackey_Glass, util
from snn_dpe.tools.network import create_network
from snn_dpe.tools.plotting import *
from snn_dpe.tools.train.timeseries import train_TS

MG_data_tr = Mackey_Glass.read_MG_data('./Data/MarkusThill-MGAB-60f6b17/1.csv', normalize=True)
MG_data_te = Mackey_Glass.read_MG_data('./Data/MarkusThill-MGAB-60f6b17/2.csv', normalize=True)

# slice up MG into input, output pairs
input_window_size = 10
output_window_size = 1

# max here is 10,000 since that's the length of the csv
n_tr_data = 2000
n_te_data = 2000

MG_inputs_tr, MG_outputs_tr = util.generate_io_pairs(MG_data_tr[:n_tr_data], input_window_size, output_window_size)
MG_inputs_te, MG_outputs_te = util.generate_io_pairs(MG_data_te[:n_te_data], input_window_size, output_window_size)


def run_test(args):
    tr_mses = []
    te_mses = []

    np.random.seed()

    noise_axis = np.linspace(0, 0.1, num=40)
    n=0
    drift_axis = np.linspace(0, 0.1, num=50)

    #for n in noise_axis:
    for d in drift_axis:

        # create a test network
        n_neurons = 30
        synapse_density = 0.1
        n_input = 12 #how many neurons will receive potentiation adjustments related to MG
        n_synapses = int((n_neurons**2)*synapse_density)

        # dpe noise
        write_noise = 0
        # synapse parameters
        drift = 0
        synapse_noise = n
        drift=d
        stdp=True

        # training parameters
        n_epochs = 5
        post_sample_reset = True #also applies to testing
        reset_synapses = False #also applies to testing
        post_epoch_reset = False
        MG_inputs_te_noise = MG_inputs_te + np.random.normal(0, 0)

        neurons = create_network(n_neurons, n_synapses, negative_weights = True, threshold_range = (0.35, 1), leak_range = (0.05, 0.25), weight_factor = 1, std_dev=synapse_noise, drift = drift, stdp=stdp)
        # train
        training_mses, testing_mses, _, _ = train_TS(n_epochs, MG_inputs_tr, MG_outputs_tr, neurons, n_input, silent=True, TS_inputs_te=MG_inputs_te_noise, TS_outputs_te=MG_outputs_te, relative=True, write_noise_std=write_noise, post_sample_reset=post_sample_reset, reset_synapses=reset_synapses, post_epoch_reset=post_epoch_reset)
        
        # save the final mses of the last epoch
        tr_mses.append(training_mses[-1])
        te_mses.append(testing_mses[-1])

    return tr_mses, te_mses, drift_axis, noise_axis

if __name__ == '__main__':

    n_tests = 10
    n_threads = 12
    with Pool(processes=n_threads) as p:
        results = list(tqdm(p.imap(run_test, range(n_tests)), total=n_tests))
        
    (tr_mses, te_mses, drift_axis, noise_axis) = np.mean(results, axis=0)



    plt.plot(drift_axis, tr_mses, label='Training')
    plt.plot(drift_axis, te_mses, label='Testing')
    plt.legend()
    plt.xlabel('Memristor conductance Drift (percentage)')
    plt.ylabel('Normalized Root Mean Squared Error')
    # plt.ylim(0, 100)
    # plt.yscale('log')
    plt.show()

    for i, r in enumerate(results):
        with open(f'MG_drift_vs_MSE_STDPsynapse_{i}.csv', 'w') as f:
            wtr = csv.writer(f, delimiter=',', lineterminator='\n')
            
            for data in zip(r[0], r[1], r[2]):
                wtr.writerow(list(data))

    all_tr_mses = []

    for r in results:
        all_tr_mses.append([])
        for d in r[1]:
            all_tr_mses[-1].append(d)
    fig = plt.subplots(figsize=(10,10), dpi=200)
    all_tr_mses = np.asarray(all_tr_mses)
    print(all_tr_mses.shape)
    print(all_tr_mses[:,1])
    plt.boxplot(all_tr_mses)
    plt.show()