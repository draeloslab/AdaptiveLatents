from adaptive_latents import datasets, proSVD, mmICA, sjPCA
from adaptive_latents.prosvd import BaseProSVD
import time
import random
import numpy as np

def a(data):
    pro = proSVD(k=10, log_level=0)
    pro.offline_run_on(data)

def b(data):
    pro = proSVD(k=10, log_level=0)
    for i in range(len(data)):
        pro.partial_fit_transform(data[i:i+1])

def c(data):
    pro = BaseProSVD(k=10)
    pro.initialize(data[:pro.k].T)
    for i in range(len(data)):
        pro.updateSVD(data[i:i + 1].T)

def d(data):
    pro = BaseProSVD(k=10)
    pro.initialize(data.T)

def e(data):
    pro = proSVD(k=10, log_level=0)
    n = 2
    data = data[: data.shape[0] - data.shape[0] % n]
    data = data.reshape((-1, n, data.shape[1]))
    pro.offline_run_on(data)

if __name__ == "__main__":
    data = datasets.Odoherty21Dataset().neural_data
    data = np.array(data)

    functions = {'a':a, 'b':b, 'c':c, 'd':d, 'e':e,}
    # functions = {'e':e}

    repeat_type = random.choice(['minimal', 1, 2, 3, 4, 5])
    sampled_function = random.choice(list(functions.keys()))


    if repeat_type == 'minimal':
        data = data[:11]
    else:
        data = np.repeat(data,repeats=repeat_type,axis=0)

    function = functions[sampled_function]

    t = time.time()
    function(data)
    t = time.time() - t
    print(t)


    import pathlib
    import json
    outfile = pathlib.Path(__file__).with_suffix(".txt")
    with open(outfile, "a+") as f:
        f.write(json.dumps(dict(
            repeat_type=repeat_type,
            sampled_function=sampled_function,
            t=t,
        )))
        f.write("\n")
