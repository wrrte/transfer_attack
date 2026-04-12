
from pathlib import Path
import torch
import random
import os

project_root = Path(os.getcwd())
# dataset path (csv & image folder directory)
ILSVRC2020_val_VT_images_path = project_root / "data" / "val_rs"
ILSVRC2020_val_VT_csv_path = project_root / "data" / "val_rs.csv"

output_image = project_root / "output_images"

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
random.seed(42)
