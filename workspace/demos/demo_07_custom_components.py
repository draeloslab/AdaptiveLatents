import numpy as np

from adaptive_latents.estimator import DecoupledEstimator, StreamingEstimator, TypicalEstimator
from adaptive_latents.tests import check_api_compatible

"""
Demo: Writing a new transformer component
"""

class FlippingEstimator0(StreamingEstimator):
    """
    StreamingTransformer is the most basic class.
    With this base class, you need to do all the logging and NaN handling manually.
    """
    def __init__(self, input_streams=None, output_streams=None, log_level=None):
        input_streams = input_streams or {0: 'X'}
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)

    def _partial_fit_transform(self, data, stream, return_output_stream):
        return (data[:, ::-1], stream) if return_output_stream else data


class FlippingTransformer1(DecoupledEstimator):
    """
    DecoupledTransformer lets you separate the fit and transform steps.
    A few transformers (like Bubblewrap and KernelSmoother) can't do this, or this would be the base class.
    """
    def __init__(self, input_streams=None, output_streams=None, log_level=None):
        input_streams = input_streams or {0: 'X'}
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)

    def _partial_fit(self, data, stream):
        pass

    def transform(self, data, stream=0, return_output_stream=False):
        if self.input_streams[stream] == 'X':
            data = data[:, ::-1]
        return (data, stream) if return_output_stream else data


class FlippingEstimator2(TypicalEstimator):
    """
    TypicalTransformer encapsulates a lot of the routing and logging information shared between transformers.
    For example, most transformers will operate on one stream, which they treat as an X variable.
    """

    def __init__(self, input_streams=None, output_streams=None, log_level=None, on_nan_width=None):
        input_streams = input_streams or {0: 'X'}
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level, on_nan_width=on_nan_width)

    def pre_initialization_fit_for_X(self, X):
        self.is_initialized = True

    def partial_fit_for_X(self, X):
        pass

    def transform_for_X(self, X):
        return X[:, ::-1]

    def inverse_transform_for_X(self, X):
        return X[:, ::-1]

    def instance_get_params(self, deep=True):
        return {}

def main():
    for FlippingTransformer in [FlippingEstimator0, FlippingTransformer1, FlippingEstimator2]:
        check_api_compatible(FlippingTransformer)

        f = FlippingTransformer()
        to_flip = np.array([[1, 2, 3], [4, 5, 6]])

        numpy_flipped = np.fliplr(to_flip)
        transformer_flipped = f.offline_run_on(to_flip)
        assert (numpy_flipped == transformer_flipped).all()


if __name__ == '__main__':
    main()