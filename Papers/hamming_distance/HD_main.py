import numpy as np
from sympy.utilities.iterables import multiset_permutations

from datetime import datetime
import json, time
import subprocess

from snn_dpe.tools.network import (create_encoders, create_network,
                                   reset_network, run_network)
from snn_dpe.tools.plotting import plot_spike_raster

from multiprocessing.pool import Pool
import tqdm
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
def run_distance_test(connectivity, n_enc = 5, n_neurons = 100, n_runs=1, save_folder = None, n_proc = 8):
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
    run['connectivity'] = connectivity
    run['n_enc'] = n_enc
    run['n_neurons'] = n_neurons
    run['n_runs'] = n_runs
    run['n_synapses'] = n_synapses
    run['min_f'] = min_f
    run['max_f'] = max_f
    run['sim_f'] = sim_f
    run['sim_time'] = sim_time

    run['enc_inputs'] = []
    run['input distances'] = []
    run['output distances'] = []
    run['input saturation 1'] = []
    run['input saturation 2'] = []
    run['output saturation 1'] = []
    run['output saturation 2'] = []

    # print(f'Running Connectivity {connectivity}...')
    if n_proc > 1 and n_runs > 1:
        args = [(n_neurons, n_synapses, n_enc, min_f, max_f, sim_f, sim_time,)] * n_runs
        with Pool(n_proc) as p:
            # run['enc_inputs'], run['input distances'], run['output distances'], \
            # run['input saturation 1'], run['input saturation 2'], run['output saturation 1'], \
            # run['output saturation 2'] = p.imap(run_distance_test_proc, args)
            rv = list(tqdm.tqdm(p.imap(run_distance_test_proc, args), total=n_runs, position=1, leave=False))

            for rv_i in rv:
                run['enc_inputs'].append(rv_i[0])
                run['input distances'].append(rv_i[1])
                run['output distances'].append(rv_i[2])
                run['input saturation 1'].append(rv_i[3])
                run['input saturation 2'].append(rv_i[4])
                run['output saturation 1'].append(rv_i[5])
                run['output saturation 2'].append(rv_i[6])

    else:
        args = (n_neurons, n_synapses, n_enc, min_f, max_f, sim_f, sim_time,)
        for _ in range(n_runs):
            sub_run = {}
            sub_run['enc_inputs'], sub_run['input distances'], sub_run['output distances'], \
            sub_run['input saturation 1'], sub_run['input saturation 2'], sub_run['output saturation 1'], \
            sub_run['output saturation 2'] = run_distance_test_proc(args)

            run['enc_inputs'].append(sub_run['enc_inputs'])
            run['input distances'].append(sub_run['input distances'])
            run['output distances'].append(sub_run['output distances'])
            run['input saturation 1'].append(sub_run['input saturation 1'])
            run['input saturation 2'].append(sub_run['input saturation 2'])
            run['output saturation 1'].append(sub_run['output saturation 1'])
            run['output saturation 2'].append(sub_run['output saturation 2'])
            
    if isinstance(save_folder, str):
        subprocess.run(['mkdir', save_folder, '-p'])
        with open(f'{save_folder}/{datetime.fromtimestamp(time.time())}', 'w') as f:
            f.write(json.dumps(run))
    else:
        return run


def run_distance_test_proc(args):
    n_neurons = args[0]
    n_synapses = args[1]
    n_enc = args[2]
    min_f = args[3]
    max_f = args[4]
    sim_f = args[5]
    sim_time = args[6]

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
    # permute encoder order (see below), simulate the network, save the spike rasters
        # this way we get a varying number of distances between each input, 
        # but the overall activity into the reservoir is the same for each run
    for enc_input in multiset_permutations(enc_inputs):
        fires = run_network(neurons, encoders, enc_input, sim_time)
        reset_network(neurons, encoders)

        network_inputs.append(fires[:, :n_enc])
        network_outputs.append(fires[:, n_enc:])

    input_distances = []
    output_distances = []
    input_saturations_1 = []
    input_saturations_2 = []
    output_saturations_1 = []
    output_saturations_2 = []
    for i in range(len(network_inputs)):
        for j in range(i + 1, len(network_outputs)):
            input_d = distance(network_inputs[i], network_inputs[j])
            output_d = distance(network_outputs[i], network_outputs[j])

            input_distances.append(input_d)
            output_distances.append(output_d)
            input_saturations_1.append(saturation(network_inputs[i]))
            input_saturations_2.append(saturation(network_inputs[j]))
            output_saturations_1.append(saturation(network_outputs[i]))
            output_saturations_2.append(saturation(network_outputs[j]))

    return list(enc_inputs), input_distances, output_distances, input_saturations_1, input_saturations_2, output_saturations_1, output_saturations_2
