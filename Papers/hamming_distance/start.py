import tqdm
from multiprocessing import Pool

from HD_main import run_distance_test

import argparse

def call_run_distance_test(args):
    return run_distance_test(*args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulates SNNs as reservoirs to determine its response to different inputs using Hamming Distance')
    parser.add_argument('save_folder', type=str,
                        help='folder to save the data')
    parser.add_argument('connectivities', metavar='N', type=float, nargs='+', default=[0.01, 0.015, 0.02, 0.09, 0.1, 0.11, 0.16, 0.18, 0.2],
                        help='a list of percentage connectivities to test [0-1]')
    parser.add_argument('n_enc', default=5, type=int,
                        help='number of encoders for the reservoir, computation time inceases exponentially with this (default: 5)')
    parser.add_argument('n_neurons', default=100, type=int,
                        help='number of neurons for the reservoir (default: 100)')
    parser.add_argument('n_runs', default=100, type=int,
                        help='number of runs to perform for each connectivity (default: 100)')
    parser.add_argument('plot', default=False, type=bool,
                        help='whether or not to plot (default: False)')

    args = parser.parse_args()

    with Pool(9) as p:
        save_folder = f'{parser.save_folder}/{parser.n_neurons}_runs_{len(parser.connectivities)}_connectivities'

        args = []
        for c in parser.connectivities:
            args.append((c, parser.n_enc, parser.n_neurons, parser.n_runs, parser.plot, save_folder))

        list(tqdm.tqdm(p.imap(call_run_distance_test, args), total=len(parser.connectivities)))