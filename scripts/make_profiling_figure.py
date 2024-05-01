import adaptive_latents as al

identifier = al.input_sources.datasets.individual_identifiers["indy"][0]
A, beh, _, _ = al.input_sources.datasets.construct_indy_data(identifier, .03)

t = al.profiling_functions.get_speed_by_time(A, beh, bw_params=dict(num=1000), max_steps=5000)

import numpy as np
import matplotlib.pyplot as plt
t *= 1000

print(np.median(np.diff(t)))
plt.plot((t[:-1] - t[0]), np.diff(t), '-')
plt.show()
