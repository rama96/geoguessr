import torch

device = torch.device("mps")
BATCH_SIZE = 128
TEST_SIZE = 0.2
OUTPUT_CLASSES = 56
EPOCHS=10
target_label_name = 'country_iso_alpha2'
model_name = "resnet_large"
experiment_name = "v3"
from_timm = True