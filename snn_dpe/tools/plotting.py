import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_network(neurons):
    # visualize
    G = nx.Graph()

    for n in neurons:
        G.add_node(n.id)

        for s in n.synapses:
            G.add_edge(n.id, s.n2.id)

    nx.draw(G)
    # plt.savefig('Images/network.pdf')
    plt.show()

# plot the data
def plot_iris_data(data, classes, attributes, labels):
    n_plots = int(len(attributes) / 2)

    fig, ax = plt.subplots(1, n_plots, figsize=(5*n_plots,5))
    
    for l, i in classes.items():
        for j, ax_i in enumerate(fig.axes):
            ax_i.scatter(data[labels==i, 2*j], data[labels==i, 2*j+1], label = l)

    for i, ax_i in enumerate(fig.axes):
        ax_i.set_xlabel(attributes[2*i])
        ax_i.set_ylabel(attributes[2*i+1])
        ax_i.legend()

def plot_spike_raster(fire_matrix, print_rates = False):
    fire_matrix = np.array(fire_matrix)
    n_neurons = len(fire_matrix[0])

    fig, ax = plt.subplots(figsize=(10,5))
    
    for i in range(n_neurons):
        f = fire_matrix[:, i]
        new_f = np.where(np.asarray(f) > 0)[0]
        ax.scatter(new_f, np.ones(len(new_f)) * i, marker='|')

    ax.set_ylim(-0.5, n_neurons - 0.5)
    ax.set_xlim(0, len(fire_matrix))
    ax.set_yticks(ticks=range(0, n_neurons+1, int(n_neurons/10)), labels=range(0, n_neurons+1, int(n_neurons/10)))
    ax.tick_params(axis='both', labelsize=20)
    ax.set_ylabel('neuron ID', fontsize=20)
    ax.set_title('Spike Raster', fontsize=20)
    ax.set_xlabel('Time (ms)', fontsize=20)

    # plt.savefig('Images/Network_output_w_early_stop.pdf')
    plt.show()

    x = np.mean(fire_matrix, axis=0)

    if print_rates:
        print('avg neuron fire rates (x):')
        for i, x_i in zip(reversed(range(n_neurons)),reversed(x)):
            print(f'neuron {i}: {x_i}')



def plot_spikes(neuron_fires, encoder_fires, attributes, einputs):
    fig, ax = plt.subplots(2, 1, figsize=(15,10), sharex=True, gridspec_kw={'height_ratios' : [1, 3]})
    
    for i, f in enumerate(encoder_fires):
        # plot encoders on separate axis
        ax[0].scatter(f, np.ones(len(f)) * i, marker='|')

    for i, f in enumerate(neuron_fires):
        ax[1].scatter(f, np.ones(len(f)) * (i), marker='|')

    enc_labels = []
    for v, ei, i in zip(attributes, einputs, range(len(encoder_fires))):
        enc_labels.append(f'{v} ({ei:.2f}) (neuron {i})')

    ax[0].set_ylim(-0.5, len(attributes) - 0.5)
    ax[0].set_yticks(ticks=range(len(attributes)), labels=enc_labels)
    ax[0].tick_params(axis='both', labelsize=15)
    ax[0].set_title('encoder spikes', fontsize=20)

    ax[1].set_ylim(-0.5, len(neuron_fires) - 0.5)
    ax[1].set_yticks(ticks=range(len(neuron_fires)), labels=range(len(neuron_fires)))
    ax[1].tick_params(axis='both', labelsize=20)
    ax[1].set_ylabel('neuron ID', fontsize=20)
    ax[1].set_title('neuron spikes', fontsize=20)


    ax[1].set_xlabel('Time (ms)', fontsize=20)

    # plt.savefig('Images/Network_output_w_enc.pdf')
    plt.show()

def plot_steady_state(steady_state_t, total_fires, window_size, m1, m2):
    # plotting code
    t = range(steady_state_t)
    plt.plot(t, total_fires)
    plt.xlabel('Time (ms)')
    plt.ylabel('Overall Network Spike Rate')
    plt.title('Spike Rate Over Time Until Steady State Reached')

    # add text and lines
    l_pos = [t[-1], t[-window_size-1], t[-window_size*2-1]] 
    tx1_pos = [l_pos[1] + window_size / 2, max(total_fires) / 2]
    tx2_pos = [l_pos[2] + window_size / 2, max(total_fires) / 2.5]
    plt.vlines(l_pos, 0, max(total_fires), colors='r', linestyles='dashed')
    plt.text(tx1_pos[0], tx1_pos[1], f'avg spikes: {m2}', horizontalalignment='center')
    plt.text(tx2_pos[0], tx2_pos[1], f'avg spikes: {m1}', horizontalalignment='center')
    # plt.savefig('Images/Spike_rate_steady_state.pdf')
    plt.show()

def plot_acc(E_t, c_acc):
    n_iter_per_epoch = int(len(c_acc) / len(E_t))

    fig = plt.subplots(figsize=(8,5))
    plt.plot(range(n_iter_per_epoch, len(c_acc)+1, n_iter_per_epoch), E_t, label='Post Epoch')
    plt.plot(range(len(c_acc)), c_acc, label='Cumulative')
    plt.xticks(ticks=range(0, (len(E_t) + 1)*n_iter_per_epoch, n_iter_per_epoch), labels=range(0, len(E_t) + 1, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    # plt.savefig('Images/Training_accuracy.pdf')
    plt.legend()
    plt.show()