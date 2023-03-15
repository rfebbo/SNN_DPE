import numpy as np


def mse(y, y_hat):
    return np.sum((y - y_hat)**2) * 1/len(y)

def rmse(y, y_hat):
    return np.sqrt(mse(y, y_hat))

# find the average spiking rate of each neuron (x) and multiply it with the dpe weights
def forward_pass(fire_matrix, dpe_weights, bias=None):
    x = np.mean(fire_matrix, axis=0)
    y = np.dot(x, dpe_weights)

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