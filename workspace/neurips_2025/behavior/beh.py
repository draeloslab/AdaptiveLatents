from Neural_Decoding import LSTMDecoder
from Neural_Decoding.preprocessing_funcs import get_spikes_with_history
from adaptive_latents import datasets
from adaptive_latents.utils import resample_matched_timeseries

if __name__ == '__main__':
    d = datasets.Odoherty21Dataset()
    beh = resample_matched_timeseries(d.behavioral_data, d.behavioral_data.t, d.neural_data.t)

    X = get_spikes_with_history(d.neural_data, bins_before=1, bins_after=0, bins_current=1)
    print(d.neural_data.shape,X.shape, beh.shape)
    model_lstm=LSTMDecoder(units=400,num_epochs=5)
    model_lstm.fit(X, beh)
    model_lstm.predict(X)
