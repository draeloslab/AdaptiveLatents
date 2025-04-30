from adaptive_latents.input_sources.autoregressor import AdamOptimizer
import jax
import numpy as np
import jax.numpy as jnp
from jax.nn import relu

def loss(s, v, lam_1=1e-3):
    u = s  # this assumes for now that the dynamics S function is an identity
    return (
            - jnp.sqrt(jnp.linalg.norm(v.T @ s))**2  # maximize dot product with the target vector
            + jnp.linalg.norm(s - v @ v.T @ s)**2  # minimize orthogonal component
            + jnp.linalg.norm(u, ord=1) * lam_1  # L1 penalty
    )

grad_loss = jax.jit(jax.value_and_grad(loss))


def design_stim(v, max_l0_norm=30, convergence_threshold=1e-2, max_outer_iters=20, max_inner_iters=250, rng=None):
    assert len(v.shape) == 2
    assert max_l0_norm > 0
    if rng is None:
        rng = np.random.default_rng()


    s_history = []
    loss_history = []

    lam_1 = 1e-3

    for _ in range(max_outer_iters):
        s = rng.uniform(size=(max(v.shape),)) * .1
        s_optimizer = AdamOptimizer(lr=0.005)

        for i in range(max_inner_iters):
            val, grad = grad_loss(s, v, lam_1=lam_1)
            s = s_optimizer.update(s,grad)
            s = relu(s)

            s_history.append(s)
            loss_history.append(val)

            if len(s_history) > 10 and jnp.linalg.norm(s_history[-2] - s_history[-1]) < convergence_threshold:
                break

        l0 = jnp.linalg.norm(s,ord=0)
        if 0 < l0 <= max_l0_norm:
            break
        if l0 == 0 or jnp.isnan(s).any():
            lam_1 /= 1.2
        else:
            lam_1 *= 2
    return np.array(s / s.max())
