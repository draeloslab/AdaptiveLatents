import os
import sys

backend = sys.argv[1]
match backend:
    case "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    case "gpu":
        pass
    case _:
        raise Exception()

offset = 0
if len(sys.argv) >= 2:
    offset = int(sys.argv[2])

import pickle
import glob
import adaptive_latents as al


def main():
    files = sorted(glob.glob(f"{al.CONFIG["output_path"]/"bubblewrap_runs"}/*.pickle"))
    brs = []
    for file in files[-2-offset:len(files)-offset]:
        with open(file, 'br') as fhan:
            brs.append(pickle.load(fhan))
    for br in brs:
        br.create_new_filenames()
        br.run(save=True, initialize=False)

if __name__ == '__main__':
    main()
