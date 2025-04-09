from adaptive_latents import datasets, proSVD, mmICA, sjPCA
from adaptive_latents.prosvd import BaseProSVD
import time
import random
import numpy as np

def a(data):
    pro = proSVD(k=10, log_level=0)
    for i in range(len(data)):
        pro.partial_fit_transform(data[i:i+1])

def b(data):
    pro = proSVD(k=10, log_level=0)
    jpca = sjPCA(log_level=0)
    for i in range(len(data)):
        o = pro.partial_fit_transform(data[i:i+1])
        jpca.partial_fit_transform(o)

def c(data):
    pro = proSVD(k=10, log_level=0)
    ica = mmICA(log_level=0)
    for i in range(len(data)):
        o = pro.partial_fit_transform(data[i:i+1])
        ica.partial_fit_transform(o)


if __name__ == "__main__":
    data = datasets.Odoherty21Dataset().neural_data
    data = np.array(data)

    functions = {'a':a, 'b':b, 'c':c}

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


    import pathlib
    import json
    outfile = pathlib.Path(__file__).with_suffix(".txt")
    with open(outfile, "a+") as f:
        to_write = json.dumps(dict(
            repeat_type=repeat_type,
            sampled_function=sampled_function,
            t=t,
        ))
        f.write(to_write)
        f.write("\n")
        print(to_write)
