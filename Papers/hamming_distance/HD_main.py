import numpy as np
from sympy.utilities.iterables import multiset_permutations

from datetime import datetime
import json, time
import subprocess

from snn_dpe.tools.network import (create_encoders, create_network,
                                   reset_network, run_network)
from snn_dpe.tools.plotting import plot_spike_raster

# measures the hamming distance between two spike rasters
def distance(sr1, sr2):
    total_distance = 0
    for n1, n2 in zip(sr1, sr2):
        for f1, f2 in zip(n1,n2):
            if f1 != f2:
                total_distance += 1

    return total_distance / sr1.size

# measures the saturation of a spike raster
def saturation(sr):
    return np.sum(sr) / sr.size

# connectivity - percentage connectivity in the reservoir
# n_enc - the total number of encoders, increasing this exponentially increases compute time
# n_nerons - the number of neurons in the reservoir
# n_runs - the number of random reservoirs created and tested
# save_folder - folder to save results to, if None don't save
# plot - will plot the spike rasters
def run_distance_test(connectivity, n_enc = 5, n_neurons = 100, n_runs=1, plot=False, save_folder = None):
    # create a test network and encoders
    n_synapses = int((n_neurons ** 2) * connectivity)

    # min and max frequency for the encoders
    min_f = 1
    max_f = 100
    # with a higher simulation frequency than the max encoder,
    # the reservoir can always perform operations between encoder inputs
    # TODO: show that ^
    sim_f = 200
    # number of timesteps at sim_f
    sim_time = 100

    # dict for storing run results
    run = {}
    run['input distances'] = []
    run['output distances'] = []
    run['input saturation 1'] = []
    run['input saturation 2'] = []
    run['output saturation 1'] = []
    run['output saturation 2'] = []
    for _ in range(n_runs):
        # seed before doing anything random in multiprocessing
        np.random.seed()
        # create random reservoir with unity thresholds, no leak, and unity synapse weights
        # this way a spike is always propagated
        neurons = create_network(n_neurons, n_synapses, threshold_range=(1, 1), leak_range=(0.0, 0.0), weight_range=(1,1))
        encoders = create_encoders(n_enc, min_f, max_f, sim_f, enc_type='frequency')

        # create random encoder inputs from no activity to full activity
        enc_inputs = np.random.uniform(0, 1, (n_enc))

        # store spike rasters for input/output pairs
        network_inputs = []
        network_outputs = []
        # permute encoder order
            # this way we get a varying number of distances between each input, 
            # but the overall activity into the reservoir is the same for each run
        for enc_input in multiset_permutations(enc_inputs):
            fires = run_network(neurons, encoders, enc_input, sim_time)
            reset_network(neurons, encoders)

            network_inputs.append(fires[:, :n_enc])
            network_outputs.append(fires[:, n_enc:])

            if plot:
                plot_spike_raster(fires[:, :n_enc], title='input')
                plot_spike_raster(fires[:, n_enc:], title='output')

        for i in range(len(network_inputs)):
            for j in range(i + 1, len(network_outputs)):
                input_d = distance(network_inputs[i], network_inputs[j])
                output_d = distance(network_outputs[i], network_outputs[j])

                run['input distances'].append(input_d)
                run['output distances'].append(output_d)
                run['input saturation 1'].append(saturation(network_inputs[i]))
                run['input saturation 2'].append(saturation(network_inputs[j]))
                run['output saturation 1'].append(saturation(network_outputs[i]))
                run['output saturation 2'].append(saturation(network_outputs[j]))

    if isinstance(save_folder, str):
        subprocess.run(['mkdir', save_folder, '-p'])
        with open(f'{save_folder}/{datetime.fromtimestamp(time.time())}', 'w') as f:
            f.write(json.dumps(run))
