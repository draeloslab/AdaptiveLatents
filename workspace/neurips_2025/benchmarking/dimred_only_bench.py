import time
from adaptive_latents import datasets
from adaptive_latents import sjPCA
import sys


if __name__ == '__main__':
    d = datasets.Odoherty21Dataset()

    n = int(sys.argv[1])
    jpca = sjPCA()
    jpca.initialize(d.neural_data[0:1, :n])
    for i in range(1,10):
        x = d.neural_data[i:i + 1, :n]
        jpca.observe(x)
    start_time = time.perf_counter_ns()
    for i in range(d.neural_data.shape[0]):
        jpca.observe(d.neural_data[i:i+1,:n])
    total_time = time.perf_counter_ns() - start_time

    import json
    import jax
    print(json.dumps({'n':n, 'total_time': total_time, 'time_of_save': time.time(), 'device': str(jax.devices()[0])}))
"""
for j in {1..50}; do
    for i in {2..20}; do
        python dimred_only_bench.py $$i >> results.txt
    done
done
"""