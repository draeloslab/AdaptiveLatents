import numpy as np
import jax
from jax import numpy as jnp
import sys
sys.path.append("/home/patriciobarber/AdaptiveLatents")

from adaptive_latents.regressions import BaseMultiKernelRegressor


class AdaptiveMultiKernelRegressor(BaseMultiKernelRegressor):
    """
    An implementation of BaseMultiKernelRegressor that uses an adaptive, 
    input-dependent length scale based on the magnitude of a flow vector.
    """
    def __init__(self, sigma_0, gamma, flow_vector_func, maxlen=100):
        """
        Initializes the adaptive regressor.

        Args:
            sigma_0 (float): The base bandwidth (sigma_0 in your formula).
            gamma (float): The scaling factor for the flow magnitude (gamma).
            flow_vector_func (callable): A function that takes an input vector 'x'
                                        and returns the flow vector 'V(x)'.
            maxlen (int): The maximum number of observations to store.
        """
        # Initialize parent with dummy length_scales since we'll compute them adaptively
        super().__init__(length_scales=(1.0,), maxlen=maxlen) 
        
        self.sigma_0 = sigma_0
        self.gamma = gamma
        self.flow_vector_func = flow_vector_func

    def make_jax_pred_f(self):
        if self.input_histories is None:
            def f(x):
                return np.array([[np.nan]])
        else:
            input_histories = [jnp.array(h) for h in self.input_histories]
            output_history = jnp.array(self.output_history)
            sigma_0 = float(self.sigma_0)
            gamma = float(self.gamma)
            
            def f(x):
                # 1. Calculate the flow vector and its magnitude for the new point 'x'
                # Note: We assume 'x' is a list of sub-vectors, so we concatenate them.
                full_x_vector = jnp.concatenate([jnp.array(sub_x).flatten() for sub_x in x])
                flow_vector = self.flow_vector_func(full_x_vector)
                flow_magnitude = jnp.linalg.norm(flow_vector)

                # 2. Calculate the adaptive bandwidth sigma(x) using your formula
                adaptive_sigma = sigma_0 / (1 + gamma * flow_magnitude)

                # 3. Convert adaptive sigma to adaptive length_scale (lambda)
                adaptive_length_scale = 1.0 / (2 * adaptive_sigma**2)

                # 4. Use this single adaptive length_scale for all input dimensions
                distances = [-adaptive_length_scale * jnp.linalg.norm(history - jnp.squeeze(jnp.array(sub_x)), axis=1) ** 2 for
                             (sub_x, history) in zip(x, input_histories)]
                
                log_weights = jnp.array(distances).sum(axis=0)
                log_weights = jnp.nan_to_num(log_weights, nan=-jnp.inf)
                log_sum = jax.scipy.special.logsumexp(log_weights)
                log_weights = log_weights - log_sum

                return jnp.exp(log_weights) @ output_history
            return f

    def predict(self, x):
        return np.array(self.make_jax_pred_f()(x))


def create_flow_vector_function():
    """
    Create a flow vector function based on the LDS dynamics.
    This is a simple example that could be replaced with a more sophisticated flow field.
    """
    def flow_vector_func(x):
        # Simple flow based on the curvy dynamics used in the LDS simulation
        # This approximates the flow field of the underlying dynamical system
        state_part = x[:3]  # Neural state dimensions
        
        # Create a flow that varies with position (similar to curvy dynamics)
        flow = jnp.array([
            -state_part[1],  # Rotational component
            state_part[0],   # Rotational component  
            0.1 * state_part[2]  # Slower dynamics in stimulation dimension
        ])
        
        return flow
    
    return flow_vector_func
