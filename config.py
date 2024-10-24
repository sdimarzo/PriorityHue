import os
from dotenv import load_dotenv

load_dotenv()

IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 100
PRETRAIN_EPOCHS = 20
LR_G = 2e-4
LR_D = 2e-4
BETA1 = 0.5
LAMBDA_L1 = 100
BETA2 = 0.999
TRAIN_PATH = os.getenv('TRAIN_PATH')
VAL_PATH = os.getenv('VAL_PATH')
PRETRAIN_MODEL_PATH = os.getenv('PRETRAIN_MODEL_PATH')
