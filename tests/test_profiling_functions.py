from adaptive_latents.profiling_functions import get_speed_by_time, get_speed_per_step
from adaptive_latents.input_sources.hmm_simulation import simulate_example_data

def test_speed_by_time():
    obs, beh = simulate_example_data(1000)
    get_speed_by_time(obs, beh)

def test_speed_per_step():
    obs, beh = simulate_example_data(1000)
    get_speed_per_step(obs, beh)
