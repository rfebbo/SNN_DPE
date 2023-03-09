import numpy as np


# This takes timeseries data and splits it into input and output pairs for training
def generate_io_pairs(timeseries_data, input_window_size, output_window_size, stride = 1):
    TS_inputs = []
    TS_outputs = []

    # iterate over the data in steps of stride while avoiding out of bounds error
    for start_idx in range(0, len(timeseries_data) - input_window_size - output_window_size, stride):

        input_i = timeseries_data[start_idx:start_idx+input_window_size]
        output_i = timeseries_data[start_idx+input_window_size:start_idx+input_window_size+output_window_size]

        TS_inputs.append(input_i)
        TS_outputs.append(output_i)

    return np.asarray(TS_inputs),np.asarray(TS_outputs)