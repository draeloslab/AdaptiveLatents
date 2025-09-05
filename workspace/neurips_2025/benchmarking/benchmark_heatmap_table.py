import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns

def load_df(file):
    df = pd.read_csv(file)
    stim_reg_updated = df['stim_reg_updated']
    df['autoreg_update'] = df.sr_update[stim_reg_updated == 0]
    df['stimreg_update']= df.sr_update[stim_reg_updated != 0]
    df = df[df.columns[np.argsort(df.max(axis=0))[::-1]]]
    df = df.loc[500:]
    return df

if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    dfs = {}
    for dim_red in ['prosvd', 'mmica', 'sjpca']:
        for pred in ['bw', 'kf', 'vjf']:
            dfs[(dim_red,pred)] = load_df(args.output.parent/f'benchmark_{dim_red}_{pred}.csv')


    mdf = pd.concat(dfs, axis=1)

    fig, axs = plt.subplots(ncols=5, layout='constrained', figsize=(18,4), sharey=True)

    for i, c in enumerate(['per_loop','stim_design', 'dimension_reduction', 'autoreg_update', 'stimreg_update']):
        means = mdf.loc[:,(slice(None), slice(None), c)].max().droplevel(2).unstack(1).T * 1000
        sns.heatmap(means, annot=True, cmap='viridis', ax=axs[i])
        axs[i].set_title(c)

    fig.savefig(args.output)