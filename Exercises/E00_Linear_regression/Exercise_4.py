# 4. Simulate more explanatory variables (*)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# a)


def simulate_x_variables(samples):
    np.random.seed(42)
    # simulate 10 0000 of each x1,x2,x3
    x1 = np.abs(np.random.normal(loc=100, scale=100, size=samples))
    x2 = np.abs(np.random.uniform(0, 50, samples))
    x3 = np.abs(np.random.normal(loc=0, scale=2, size=samples))
    epsilon = np.random.normal(loc=0, scale=50, size=samples)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    df["y"] = 25 + 2 * x1 + 0.5 * x2 + 50 * x3 + epsilon
    df["ones"] = 1
    return df


def plot_histograms(data, head_title):
    fig, ax = plt.subplots(2, 2, dpi=100, figsize=(16, 8))

    ax[0,0].hist(data["x1"])
    ax[0,1].hist(data["x2"])
    ax[1,0].hist(data["x3"])
    ax[1,1].hist(data["y"])
    fig.suptitle(head_title, size=18)
    ax[0,0].set(ylabel="Frequency")
    ax[0,0].set_title("Minutes")
    ax[0,1].set(ylabel="Frequency")
    ax[0,1].set_title("SMS")
    ax[1,0].set(ylabel="Frequency")
    ax[1,0].set_title("Surf (GB)")
    ax[1,1].set( ylabel="Frequency")
    ax[1,1].set_title("Cost")
    plt.show()


def start_script():
    df = simulate_x_variables(10000)
    plot_histograms(df, "Histogram with constraint line")

    df_outliers_rem = df[(df["x1"] < 300) & (df["x3"] < 4) & (df["y"] > 0)]
    plot_histograms(df_outliers_rem, "Histogram with outliers removed")


#start_script()  # remove # if run start_script which make exercise 4 to run


