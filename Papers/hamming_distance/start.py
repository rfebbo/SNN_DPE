import tqdm
from multiprocessing import Pool
import numpy as np

from HD_main import run_distance_test

import argparse

def call_run_distance_test(args):
    return run_distance_test(*args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulates SNNs as reservoirs to determine its response to different inputs using Hamming Distance')
    parser.add_argument('--save_folder', type=str,
                        help='folder to save the data')
    parser.add_argument('--n_proc', type=int, default=8,
                        help='number of processes')
    parser.add_argument('--c_min', type=float, default=0.01, required=False,
                        help='start of connectivities to test [0-1]')
    parser.add_argument('--c_max', type=float, default=0.2, required=False,
                        help='end of connectivities to test [0-1]')
    parser.add_argument('--c_step', type=float, default=0.01, required=False,
                        help='step of connectivities to test [0-1]')
    parser.add_argument('--n_enc', default=5, type=int, required=False,
                        help='number of encoders for the reservoir, computation time inceases exponentially with this (default: 5)')
    parser.add_argument('--n_neurons', default=100, type=int, required=False,
                        help='number of neurons for the reservoir (default: 100)')
    parser.add_argument('--n_runs', default=100, type=int, required=False,
                        help='number of runs to perform for each connectivity (default: 100)')
    parser.add_argument('--plot', default=False, type=bool, required=False,
                        help='whether or not to plot (default: False)')

    args = parser.parse_args()

    connectivities = np.arange(args.c_min, args.c_max, args.c_step)
    save_folder = f'{args.save_folder}/{args.n_runs}_runs_{len(connectivities)}_connectivities'
    pool_args = []
    for c in connectivities:
        pool_args.append((c, args.n_enc, args.n_neurons, args.n_runs, args.plot, save_folder))

    if args.n_proc > 1:
        with Pool(args.n_proc) as p:
            list(tqdm.tqdm(p.imap(call_run_distance_test, pool_args), total=len(connectivities)))
    else:
        for a in tqdm.tqdm(pool_args, total=len(pool_args)):
            call_run_distance_test(a)
    
