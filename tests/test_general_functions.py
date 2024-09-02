import adaptive_latents
import jax


class TestEnvironment:
    def test_can_use_configured_backend(self):
        # note that this does not check that both backends are possible
        from jax.lib import xla_bridge
        assert xla_bridge.get_backend().platform == adaptive_latents.CONFIG["jax_platform_name"]

    def test_can_use_float64(self):
        jax.config.update('jax_enable_x64', True)  # this line is for documentation, the real line is in the config load
        x = jax.random.uniform(jax.random.key(0), (1,), dtype=jax.numpy.float64)
        assert x.dtype == jax.numpy.float64

    # for documentation purposes
    """
    def test_can_use_float32(self):
        import jax
        # jax.config.update('jax_enable_x64', False)
        x = jax.random.uniform(jax.random.key(0), (1,), dtype=jax.numpy.float64)
        assert x.dtype != jax.numpy.float64
    """

# class TestBWRun:
#     def test_post_hoc_regression_is_correct(self, rng, outdir):
#         m, n_obs, n_beh = 150, 3, 4
#         obs = rng.normal(size=(m, n_obs))
#         beh = rng.normal(size=(m, n_beh))  # TODO: refactor these names to input and output
#         t = np.arange(m)
#
#         br1 = BWRun(
#             Bubblewrap(3, **Bubblewrap.default_clock_parameters),
#             NumpyTimedDataSource(obs, t, (0, 1)),
#             NumpyTimedDataSource(beh, t, (0, 1)),
#             SemiRegularizedRegressor(Bubblewrap.default_clock_parameters['num'], n_beh),
#             show_tqdm=False,
#             output_directory=outdir
#         )
#         br1.run()
#
#         br2 = BWRun(Bubblewrap(3, **Bubblewrap.default_clock_parameters), NumpyTimedDataSource(obs, t, (0, 1)), show_tqdm=False, output_directory=outdir)
#         br2.run()
#         reg = SemiRegularizedRegressor(br2.bw.N, n_beh)
#         br2.add_regression_post_hoc(reg, NumpyTimedDataSource(beh, t, (0, 1)))
#
#         assert np.allclose(br1.h.beh_error[1], br2.h.beh_error[1], equal_nan=True)
#         # TODO: the times seem to be 1 out of sync with the expected bubblewrap step times (even accounting for delay)



# TODO:
#  array shapes are correct for 1d output
#  test different regressors work together
#  test_can_save_and_reload
#  test_nsteps_inbwrun_works_correctly
#  also should make the timing of logs more clear
#  make sure the config in-file defaults equal the repo defaults
