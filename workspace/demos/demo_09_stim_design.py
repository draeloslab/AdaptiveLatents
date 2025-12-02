from adaptive_latents import StreamingKalmanFilter, StimRegressor, BaseMultiKernelRegressor, ArrayWithTime
from adaptive_latents.stim_designer import StimDesigner
from adaptive_latents.input_sources.lds_simulation import LDS
import numpy as np
import jax.numpy as jnp
import jax

import datetime
import uuid
from pathlib import Path

def get_or_create_session_id(folder_path, seed_num):
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    session_file = folder / "session_id.txt"

    # If session already exists → reuse
    if session_file.exists():
        return session_file.read_text().strip()

    # Otherwise create a new one
    new_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed_num}"
    session_file.write_text(new_id)

    return new_id

"""
Demo: Stimulus design
"""

def make_stims_3d(stimulations, rng): # TODO: move this to lds_simulation
    def angle_beween(v1, v2):
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        return angle

    new_stimulations = []
    for i, stimulation in enumerate(stimulations):
        new_stimulation = np.zeros(3)
        if stimulation.any():
            new_stimulation[2] = stimulation[0]
        elif i > 0 and stimulations[i - 1].any():
            a_between = 0
            while a_between < 5 * np.pi / 180:
                new_stimulation = rng.normal(size=(3,))
                new_stimulation = new_stimulation / np.linalg.norm(new_stimulation)
                new_stimulation = np.abs(new_stimulation)
                a_between = angle_beween(new_stimulation, np.array([0, 0, 1]))
        new_stimulations.append(ArrayWithTime(new_stimulation, stimulation.t))
    return ArrayWithTime.from_list(new_stimulations)




def main(show_plots=True):
    jax.config.update("jax_platform_name", "cpu")
    SEED = 135978
    #170934
    #98247
    #39428
    #24509824
    #7982

    # ---- SESSION HANDLING ----
    SESSION_FOLDER = r"C:\Users\secom\OneDrive\Documents\DraelosLab\AdaptiveLatents\AdaptiveLatents\workspace\Alexworkspace\metricsfolder"

    session_id = get_or_create_session_id(SESSION_FOLDER, SEED)      # persists across runs
    script_run_id = uuid.uuid4().hex            # new every time you run main.py

    print(f"Session ID: {session_id}")
    print(f"Script run ID: {script_run_id}")


    for iter in range(3):
        _, observations, stimulations = LDS.run_nest_dynamical_system(rotations=100, u_function='curvy', 
                                                                      rng=SEED
                                                                      ) #seed r g
        # look into observationd matrix
        # kernel parameters (for the kernel regression) tweaking 
        # see if multiple optim runs all arrive at similar minimum and compare
        # 

        stimulations = make_stims_3d(stimulations, np.random.default_rng(SEED))


        sr = StimRegressor(
            autoreg=StreamingKalmanFilter(steps_between_refits=5),
            stim_reg=BaseMultiKernelRegressor(
                length_scales=[0.1, 2.0,
                               0.005],
                maxlen=100
                ),
        )

        sr.offline_run_on([(observations, 'X'),  (stimulations, 'stim')])



        # goal = np.zeros((3,1))
        # goal[0] = 0
        # goal[1] = 1/np.sqrt(2)
        # goal[2] = 1/np.sqrt(2)
        # seed 10 with goal[1] = 1
        goallist = np.array([np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1]),
                             #np.array([1/np.sqrt(2),1/np.sqrt(2),0]),np.array([1/np.sqrt(2),0,1/np.sqrt(2)]),np.array([0,1/np.sqrt(2),1/np.sqrt(2)]),
                             #np.array([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)]),
                             #np.array([np.sqrt(3)/2,0.5,0]),np.array([0.5,np.sqrt(3)/2,0]),np.array([0.5,0,np.sqrt(3)/2]),
                             #np.array([np.sqrt(3)/2,0,0.5]),np.array([0,0.5,np.sqrt(3)/2]),np.array([0,np.sqrt(3)/2,0.5])
                             ])
        #goallist = np.array([0,0,1])

        # stim_designer_open_loop = StimDesigner(optimization_method='jaxopt', should_log=True)
        # u = stim_designer_open_loop.design_stim(
        #     goal,
        #     u_dimension=3,
        #     u_to_s_function=lambda u: u,
        # )
        # print(u)
        # check out `stim_designer_open_loop.log`

        stim_designer_closed_loop = StimDesigner(optimization_method='admm',
                                                rng_seed = SEED,
                                                session_id=session_id,              
                                                script_run_id=script_run_id,      
                                                log_dir=SESSION_FOLDER,
                                                )
        
        #regressed_u_to_s_function = lambda u: sr.stim_reg.make_jax_pred_f()([np.array([20, 0, 10]), u , observations.t[-1]]) # we want time to matter less
        #regressed_u_to_s_function = sr.stim_reg.make_jax_pred_f()

#         f_pred = sr.stim_reg.make_jax_pred_f()
#         def regressed_u_to_s_function(u):
#             return f_pred([jnp.array([20, 0, 10]),u, jnp.array([float(observations.t[-1])]) ])
# # [-15,0,10]
        # IH = [np.asarray(h) for h in sr.stim_reg.input_histories]
        # print("num_streams:", len(IH))
        # for i, h in enumerate(IH): print(f"stream {i} shape:", h.shape)
        # print("len(length_scales) =", len(sr.stim_reg.length_scales))
        
        rng0 = np.random.default_rng(SEED)
        u0 = rng0.uniform(size=(observations.shape[1],)) * 0.2
        #u0 = [0.1,0.1,0.1,0.1]

        f_pred = sr.stim_reg.make_jax_pred_f()
        
        def regressed_u_to_s_function(u):
            return jnp.ravel(
                f_pred([
                    jnp.array([20,0,10]), 
                    jnp.asarray(u).reshape(observations.shape[1],), 
                    jnp.array([float(observations.t[-1])])
                ])
            )

        # choose the *same* u0 you feed to the optimizer
        u0 = np.asarray(u0)   # whatever your init_u is

        u = stim_designer_closed_loop.design_stim(
            goallist[iter][:, None],
            #goallist[:,None],
            u_dimension=observations.shape[1],
            u_to_s_function= regressed_u_to_s_function,
            init_u = u0
        )
        print(u) # return u at the end of design stim is returned here


if __name__ == '__main__':
    main()


# heat map for the loss landscape to visualize how im doing it



# closed loop working: 229-244, 281-294, 411-424

# starting 762 is only open loop

#1111