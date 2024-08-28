from adaptive_latents import CONFIG, AnimationManager, Pipeline, CenteringTransformer, proSVD, sjPCA
from adaptive_latents.jpca import generate_circle_embedded_in_high_d
import numpy as np


def main(output_directory=CONFIG['plot_save_path'], steps_to_run=None):
    rng = np.random.default_rng(0)
    input_data, _, _ = generate_circle_embedded_in_high_d(rng, m=steps_to_run)

    pro = proSVD(k=4)

    p = Pipeline([
        CenteringTransformer(),
        pro,
        sjPCA()
    ])

    with AnimationManager(fps=20, outdir=output_directory) as am:
        transformed_data = np.zeros((input_data.shape[0], pro.k))

        for i, output in enumerate(p.run_on(input_data)):
            if np.isnan(output).any():
                transformed_data[i] = np.nan
                continue

            transformed_data[i] = output

            am.axs[0, 0].cla()
            am.axs[0, 0].scatter(transformed_data[:i, 0], transformed_data[:i, 1])
            am.grab_frame()



if __name__ == '__main__':
    main(output_directory='.', steps_to_run=500)
