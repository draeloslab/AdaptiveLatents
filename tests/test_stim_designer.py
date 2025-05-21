from adaptive_latents.stim_designer import StimDesigner
from adaptive_latents.regressions import BaseKernelRegressor
import numpy as np
import jax
import matplotlib.pyplot as plt

def test_design_stim(rng: np.random.Generator):
    for max_nonzero_elements in [1, 5, 10, 100]:
        sd = StimDesigner(max_l0_norm=max_nonzero_elements,)
        v = rng.normal(size=(100,1))
        v = v/np.linalg.norm(v)
        s, _ = sd.design_stim(v, u_dimension=100)
        assert np.linalg.norm(s, ord=0) <= max_nonzero_elements

def test_kernel_regression_integration(show_plots):
    sd1 = StimDesigner(max_l0_norm=1)
    sd2 = StimDesigner(max_l0_norm=1)

    reg = BaseKernelRegressor(length_scale=20)

    theta_shift = np.pi/2
    train_xs = []
    for theta in np.linspace(0, 2*np.pi, 100):
        x = np.array([np.cos(theta), np.sin(theta)])
        y = np.array([np.cos(theta + theta_shift), np.sin(theta + theta_shift)])
        reg.observe(x=x, y=y)
        train_xs.append(x)
    train_xs = np.array(train_xs)

    target = np.array([[0,1]]).T
    if show_plots:
        ####################################################################################################################
        #
        # fig, ax = plt.subplots()
        # for theta in np.linspace(0, 2*np.pi, 50):
        #     x = np.array([np.cos(theta), np.sin(theta)])
        #     y = np.array([np.cos(theta + theta_shift), np.sin(theta + theta_shift)])
        #     pred = reg.predict(x)
        #     ax.plot([x[0], y[0]], [x[1], y[1]], 'k-', alpha=.25)
        #     ax.plot([x[0], pred[0]], [x[1], pred[1]])
        #
        # ax.plot(train_xs[:,0], train_xs[:,1], color='k')
        # ax.axis('equal')
        # plt.show(block=True)

        ####################################################################################################################
        fig, ax = plt.subplots()

        # u_to_s_function = lambda x: x
        u_to_s_function = reg.make_jax_pred_f()

        def inner_loss(x):
            return jax.numpy.linalg.norm(u_to_s_function(x) - target[:,0]) ** 2
        xs = np.linspace(-1,1,99)
        ys = np.linspace(-1,1,100)
        grad = jax.jit(jax.grad(inner_loss))
        result = np.zeros((len(ys), len(xs), 2))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                result[j, i, :] = grad(np.array([x,y]))
        result = -np.array(result)
        ax.streamplot(xs,ys, result[...,0], result[...,1])
        ax.plot(train_xs[:,0], train_xs[:,1], color='k')
        ax.scatter(target[0], target[1], c='r')
        ax.axis('equal')
        plt.show(block=True)

    assert np.allclose(target.flatten(), sd1.design_stim(target, u_dimension=2)[0])
    assert np.allclose(target.flatten()[::-1], sd2.design_stim(target, u_dimension=2, u_to_s_function=reg.make_jax_pred_f())[0])
