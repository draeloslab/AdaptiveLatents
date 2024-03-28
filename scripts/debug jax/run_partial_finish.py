import os
import sys
from jax.config import config
config.update("jax_enable_x64", True)

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
        # br.bw.m_L_diag = 0 * br.bw.m_L_diag
        # br.bw.v_L_diag = 0 * br.bw.v_L_diag
        # br.bw.m_L_lower = 0 * br.bw.m_L_lower
        # br.bw.v_L_lower = 0 * br.bw.v_L_lower
        br.run(save=True, initialize=False)

if __name__ == '__main__':
    main()
