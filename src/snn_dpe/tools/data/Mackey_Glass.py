import numpy as np


def read_MG_data(filename, normalize=True):
    data = []
    with open(filename, 'r') as f:
        for i, l in enumerate(f):
            if i > 0:
                data.append(np.asarray(l.split(','), dtype=np.float64))


    data = np.asarray(data)
    # the actual values are in the second column
    data = data[:,1]
    
    if normalize:
        # maybe we don't need to do this?
        # I don't like this but lets normalize the dataset to values from 0 to 1
        dmin = np.min(data)
        dmax = np.max(data)
        normalized_data = []
        for d in data:
            v = (d - dmin) / (dmax - dmin)
            normalized_data.append(v)

        data = np.asarray(normalized_data)

    return data