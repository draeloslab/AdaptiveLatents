import os
import sys

match sys.argv[1]:
    case "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    case "gpu":
        pass
    case _:
        raise Exception()

from jax.lib import xla_bridge
from adaptive_latents.bubblewrap import *


bw = Bubblewrap(dim=2, num=5)

N = 50
t = numpy.arange(0, 100, .5)
X = numpy.column_stack([numpy.cos(t), numpy.sin(t)])

for i in range(0,bw.M):
    bw.observe(X[i])

bw.init_nodes()

print(f"{numpy.array(bw.L)[0][0][0]:.32f} ({ xla_bridge.get_backend().platform })")