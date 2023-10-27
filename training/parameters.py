import os

# DCASE paths
dcase_2017_dev_path = "/workspace/noise_databases/dcase/tut_2017_development/TUT-acoustic-scenes-2017-development/"
dcase_2018_dev_path = "/workspace/noise_databases/dcase/tut_2018_development_mobile/TUT-urban-acoustic-scenes-2018-mobile-development/"
dcase_2020_dev_path = "/workspace/noise_databases/dcase/tut_2020_development_mobile/TAU-urban-acoustic-scenes-2020-mobile-development/"
dcase_2017_eval_path = "/workspace/noise_databases/dcase/tut_2017_evaluation/"
dcase_2018_eval_path = "/workspace/noise_databases/dcase/tut_2018_evaluation_mobile/TUT-urban-acoustic-scenes-2018-mobile-evaluation/"
dcase_2020_eval_path = "/workspace/noise_databases/dcase/tut_2020_evaluation_mobile/TAU-urban-acoustic-scenes-2020-mobile-evaluation/"


# Training parameters
MODEL = "unet"
WAVEFORM_SAMPLING_RATE = 8000
DURATION = 3
N_SEGMENTS = 5
RUN_VAL = True
BATCH_SIZE = 128
TRAIN_STEPS = 64
TRAIN_BUFFER_SIZE = TRAIN_STEPS * BATCH_SIZE
VAL_STEPS = 64
VAL_BUFFER_SIZE = VAL_STEPS * BATCH_SIZE
LEARNING_RATE = 1e-3
PATIENCE = 10
FACTOR = 0.1
EARLY_STOP = 20
MIN_DELTA = 0
NB_EPOCHS = 500
FACTOR_SC = 0.5
FACTOR_MAG = 0.5
CKPT_NAME = f"{MODEL}_lr_{LEARNING_RATE}_BS_{BATCH_SIZE}"
CKPT_PATH = f"{os.path.dirname(__file__)}/checkpoints/{CKPT_NAME}/"
