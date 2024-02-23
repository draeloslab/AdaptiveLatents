import os
import sys

match sys.argv[1]:
    case "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    case "gpu":
        pass
    case _:
        raise Exception()

import pickle
import glob
import adaptive_latents as al


def main():
    files = sorted(glob.glob(f"{al.CONFIG["output_path"]/"bubblewrap_runs"}/*.pickle"))
    brs = []
    for file in files[-2:]:
        with open(file, 'br') as fhan:
            brs.append(pickle.load(fhan))
    for br in brs:
        print(br.frozen)
        br.run(save=True)

if __name__ == '__main__':
    main()




