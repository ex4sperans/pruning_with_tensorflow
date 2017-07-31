from matplotlib import pyplot as plt

def plot_histogram(weights_list, image_name, include_zeros=True):

    weights = []
    for w in weights_list:
        weights.extend(list(w.ravel()))

    if not include_zeros:
        weights = [w for w in weights if w != 0]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    ax.hist(weights,
            bins=100,
            facecolor='green',
            edgecolor='black',
            alpha=0.7,
            range=(-0.1, 0.1))

    ax.set_title('Weights distribution')

    fig.savefig(image_name + '.png')