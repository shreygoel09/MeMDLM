from transformers.optimization import get_constant_schedule_with_warmup

MODE = "train" # train, ppl_eval, sample_eval
DIFFUSION = "absorbing_state"
BACKBONE = "dit"
PARAMETERIZATION = "subs" # subs, d3pm, sedd
TIME_CONDITIONING = False
T = 0 # or 1000
SUBS_MASKING = False
LR_SCHEDULER = get_constant_schedule_with_warmup

LATENT_DIM = 1280
MODEL_NAME = 'facebook/esm2_t33_650M_UR50D'
ESM_LAYERS = 3

SEED = 1

class Loader:
    BATCH_SIZE = 2
    DATA_PATH = "/workspace/a03-sgoel/MDpLM/data"
    
class Sampling:
    PREDICTOR = "ddpm_cache"  # analytic, ddpm, ddpm_cache (recommended)
    STEPS = 128
    NOISE_REMOVAL = True
    STRIDE_LENGTH = 1
    NUM_STRIDES = 1
    NUM_SAMPLING_BATCHES = 2
    NUM_SAMPLE_LOG = 2

class Model:
    hidden_size = 1280
    cond_dim = 256
    n_heads = 8
    n_blocks = 2
    dropout = 0.5
    length = 512

class Training:
    EMA = 0.9999
    ANTITHETIC_SAMPLING = True
    IMPORTANCE_SAMPLING = False
    CHANGE_OF_VARIABLES = False
    SAMPLING_EPS = 1e-3
    ACCUMULATE_GRAD_BATCHES = 2
    NUM_EPOCHS = 2
    GRADIENT_CLIP_VAL = 1.0
    PRECISION = 'bf16'
    MAX_STEPS = 1,000,000
    LOG_EVERY_N_STEPS: 10
    GPUS = 2
    SAVE_DIR = "/workspace/a03-sgoel/MDpLM/models"
    MLM_MODEL_PATH = "/workspace/a03-sgoel/MDpLM/benchmarks/MLM/model_ckpts/best_model_epoch"

class Eval:
    CHECKPOINT_PATH = "/workspace/a03-sgoel/MDpLM/checkpoints"
    DISABLE_EMA = False
    COMPUTE_GENERATIVE_PERPLEXITY = False
    COMPUTE_PERPLEXITY_ON_SANITY = False
    PERPLEXITY_BATCH_SIZE = 8
    GENERATE_SAMPLES = True

class Optim:
    LR = 1e-4
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1e-8
    WEIGHT_DECAY = 0
    NUM_WARMUP_STEPS = 3

class Noise:
    NOISE_TYPE = 'loglinear'

# finish this when training actually works
class Wandb:
    PROJECT = "MDpLM_shrey_test"
    GROUP = "programmablebio"