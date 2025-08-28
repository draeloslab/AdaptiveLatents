from benchmark_heatmap_table import load_df
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_csv", type=pathlib.Path, required=True)
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    df = load_df(args.input_csv)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), layout='constrained')
    axs[0].plot(df.per_loop)
    axs[0].set_title('Per iteration time over time')
    axs[0].set_xlabel('timepoints')
    axs[0].set_ylabel('time (s)')

    axs[1].hist(df.per_loop, bins=100)
    axs[1].semilogy()
    axs[1].set_title('Per iteration time histogram')
    axs[1].set_xlabel('processing time')
    axs[1].set_ylabel('time (s)')

    fig.savefig(args.output, bbox_inches="tight")

