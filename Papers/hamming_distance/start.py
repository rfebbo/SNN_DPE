import tqdm
from multiprocessing import Pool

from HD_main import run_distance_test

def call_run_distance_test(args):
    return run_distance_test(*args)

# connectivity, n_enc = 5, n_neurons = 100, n_runs=1, plot=False, save_folder = None):
if __name__ == '__main__':
    with Pool(9) as p:
        connectivities = [0.01, 0.015, 0.02, 0.09, 0.1, 0.11, 0.16, 0.18, 0.2]
        n_enc = 5
        n_neurons = 100
        n_runs = 100
        plot = False
        save_folder = f'{n_neurons}_runs_{len(connectivities)}_connectivities'

        args = []
        for c in connectivities:
            args.append((c, n_enc, n_neurons, n_runs, plot, save_folder))

        list(tqdm.tqdm(p.imap(call_run_distance_test, args), total=len(connectivities)))