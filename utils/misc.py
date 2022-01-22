import os
from datetime import datetime
from pathlib import Path

import arviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpyro.diagnostics import summary

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

def make_output_folder():

    folder_name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    path = Path("./output/", folder_name)
    os.makedirs(path)
    
    return path


def mcmc_summary_to_dataframe(mcmc):

    # Create summary dictionary

    summary_dict = summary(mcmc._states["z"])

    # Create archetype dataframe

    df = pd.DataFrame(columns=list(summary_dict[list(summary_dict.keys())[0]]))

    # Parse out the key entries

    var_name = []
    mean = np.array([])
    std = np.array([])
    median = np.array([])
    p5 = np.array([])
    p95 = np.array([])
    n_eff = np.array([])
    r_hat = np.array([])

    for k, v in summary_dict.items():

        if isinstance(list(summary_dict[k].values())[0], np.ndarray):
            total = len(list(summary_dict[k].values())[0])
        else:
            total = 1

        modified_names = [k + "_" + str(n) for n in range(0, total)]
        var_name = [*var_name, *modified_names]

        for i, (item, value) in enumerate(summary_dict[k].items()):
            if item == "mean":
                mean = np.append(mean, value)
            elif item == "std":
                std = np.append(std, value)
            elif item == "median":
                median = np.append(median, value)
            elif item == "5.0%":
                p5 = np.append(p5, value)
            elif item == "95.0%":
                p95 = np.append(p95, value)
            elif item == "n_eff":
                n_eff = np.append(n_eff, value)
            elif item == "r_hat":
                r_hat = np.append(r_hat, value)

    # Append to the dataframe

    df["mean"] = mean
    df["std"] = std
    df["median"] = median
    df["5.0%"] = p5
    df["95.0%"] = p95
    df["n_eff"] = n_eff
    df["r_hat"] = r_hat
    # df.index = var_name
    
    return df


def plot_extra_fields(mcmc, output_path):

    extra_fields = mcmc.get_extra_fields(group_by_chain=False)
    
    # Acceptance probability
    
    plt.figure(figsize=(15, 7))
    plt.scatter(x=np.arange(0, len(extra_fields['accept_prob'])), 
                y=extra_fields['accept_prob'], 
                c=["red"], alpha=0.5, edgecolors="face", linewidths=2)
    plt.title("Acceptance Probability of Proposal")
    plt.grid()
    plt.savefig(Path(output_path, 'accept_prob.jpg'), transparent=False)
    plt.show()
    
    # Steps in HMC trajectory
    
    plt.figure(figsize=(15, 7))
    plt.scatter(x=np.arange(0, len(extra_fields['num_steps'])), 
                y=extra_fields['num_steps'], 
                c=["green"], alpha=0.5, edgecolors="face", linewidths=2)
    plt.title("Number of steps in HMC trajectory")
    plt.grid()
    plt.savefig(Path(output_path, 'hmc_steps.jpg'), transparent=False)
    plt.show()

def print_extra_fields(mcmc):

    extra_fields = mcmc.get_extra_fields(group_by_chain=False)
    df = pd.DataFrame()
    df["accept_prob"] = extra_fields['accept_prob']
    df["num_steps"] = extra_fields['num_steps']
    print(df)

def plot_traces(var, output_path):

    mcmc_obj = arviz.from_numpyro(posterior=mcmc)
    arviz.plot_trace(mcmc_obj, var_names=var, filter_vars="like", 
                     kind="trace", compact=True, figsize=(20, 7));
    plt.grid()
    plt.savefig(Path(output_path, '{}.jpg'.format(var)), transparent=False)
    plt.show()


def rhat_histogram(df, output_path):

    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.hist(df["r_hat"], bins=50, color="red", density=True, stacked=True);
    plt.savefig(Path(output_path, 'rhat_distribution.jpg'), transparent=False)
    plt.show()