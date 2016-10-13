import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py

HOME_PATH = os.path.expanduser('~')


def main():
    topic = pd.read_pickle('../pickles/topic_0.pickle')
    topic = topic.to_dict()

    # print(topic)
    print np.random.randn(1000)
    exit()

    plt.hist(topic.items())
    plt.title("Gaussian Histogram")
    plt.xlabel("Coordinate")
    plt.ylabel("% Distribution")

    fig = plt.gcf()

    plot_url = py.plot_mpl(fig, filename='mpl-basic-histogram')

if __name__ == '__main__':
    main()
