# ## Parameters
# N = 1000             # number of nodes to tile with
# lam = 1e-3          # lambda
# nu = 1e-3           # nu
# eps = 1e-3          # epsilon sets data forgetting
# step = 8e-2         # for adam gradients
# M = 30              # small set of data seen for initialization
# B_thresh = -10      # threshold for when to teleport (log scale)
# batch = False       # run in batch mode
# batch_size = 1      # batch mode size; if not batch is 1
# go_fast = False     # flag to skip computing priors, predictions, and entropy for optimal speed
 # num_grad_q = 1     # number of times to call grad_q in an inner loop

default_rwd_parameters = dict(
    num=200,
    lam=1e-3,
    nu=1e-3,
    eps=1e-3,
    step=8e-2,
    M=30,
    B_thresh=-10,
    batch=False,
    batch_size=1,
    go_fast=False,
    seed=42,
    num_grad_q=1,
    copy_row_on_teleport=True,
    sigma_orig_adjustment=0,
    n_thresh=5e-4,
)

default_jpca_dataset_parameters = dict(
    default_rwd_parameters,
    num=60,
    eps=3.360463736782607e-07,
    step=8e-2,
    M=30,
    B_thresh=-6.59005629959783,
    batch=False,
    batch_size=1,
    go_fast=False,
    seed=42,
    num_grad_q=1,
    copy_row_on_teleport=True,
    sigma_orig_adjustment=0,
    n_thresh=5e-4,
)

default_clock_parameters = dict(
    num=8,
    lam=1e-3,
    nu=1e-3,
    eps=1e-4,
    step=8e-2,
    M=100,
    B_thresh=-5,
    batch=False,
    batch_size=1,
    go_fast=False,
    seed=42,
    num_grad_q=1,
    copy_row_on_teleport=True,
    sigma_orig_adjustment=0,
    n_thresh=5e-4,
)

# reasonable_parameter_ranges = dict(
#     ?
# )
