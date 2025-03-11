from adaptive_latents import ArrayWithTime, CenteringTransformer, Concatenator, KernelSmoother, Pipeline, datasets, proSVD, sjPCA

"""
Demo: Manually managing data flow
"""

def main():
    d = datasets.Odoherty21Dataset()
    neural_data = d.neural_data
    behavioral_data = d.behavioral_data

    p = Pipeline([
        centerer := CenteringTransformer(input_streams={0: 'X'}),
        smoother := KernelSmoother(input_streams={0: 'X'}),
        concat := Concatenator(input_streams={0: 0, 1: 1}, output_streams={0: 0, 1: 0}),
        pro := proSVD(k=6),
        jpca := sjPCA(),
    ])

    outputs = {}
    # streaming_run_on has mostly the same syntax as offline_run_on
    for data, stream in Pipeline().streaming_run_on([neural_data, behavioral_data], return_output_stream=True):
        if data.t > 60:
            break
        data, stream = centerer.partial_fit_transform(data, stream, return_output_stream=True)

        # you can do what you want between steps, but be careful, it can be hard to reason about
        # data = data + 1
        # stream = 1-stream

        data, stream = smoother.partial_fit_transform(data, stream, return_output_stream=True)
        data, stream = concat.partial_fit_transform(data, stream, return_output_stream=True)
        data, stream = pro.partial_fit_transform(data, stream, return_output_stream=True)
        data, stream = jpca.partial_fit_transform(data, stream, return_output_stream=True)

        if stream not in outputs:
            outputs[stream] = []
        outputs[stream].append(data)

    manual_latents = ArrayWithTime.from_list(outputs[0], squeeze_type='to_2d', drop_early_nans=True)

    # p = p.blank_copy() # TODO: make this work
    p = Pipeline([x.blank_copy() for x in p.steps])
    auto_latents = p.offline_run_on([neural_data, behavioral_data], exit_time=60)
    assert (manual_latents == auto_latents).all()


if __name__ == '__main__':
    main()