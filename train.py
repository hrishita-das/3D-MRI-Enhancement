import os
import torch
from torch.utils.data import DataLoader

from utils.seed import set_seed
from utils.config import load_config

from datasets.transforms import MRITransform
from datasets.ixi_dataset import IXIUnpairedDataset
from datasets.pmc_dataset import PMCPairedDataset

from models.unet3d import UNet3D
from losses.reconstruction import ReconstructionLoss
from trainers.trainer import Trainer


# -------------------------------------------------------
# Load configs
# -------------------------------------------------------

train_cfg = load_config("configs/train.yaml")
dataset_cfg = load_config("configs/dataset.yaml")
model_cfg = load_config("configs/model.yaml")

# -------------------------------------------------------
# Seed
# -------------------------------------------------------

set_seed(train_cfg["seed"])

# -------------------------------------------------------
# Device
# -------------------------------------------------------

if torch.cuda.is_available():

    device = torch.device(train_cfg["device"])

    torch.cuda.set_device(device)

else:

    device = torch.device("cpu")


print("=" * 60)
print("CUDA_VISIBLE_DEVICES :", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("PyTorch CUDA devices :", torch.cuda.device_count())

if torch.cuda.is_available():

    print("Current CUDA Device :", torch.cuda.current_device())
    print("GPU Name            :", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Training Device      :", device)
print("=" * 60)


# -------------------------------------------------------
# Transforms
# -------------------------------------------------------

train_transform = MRITransform(training=True)
val_transform = MRITransform(training=False)

# -------------------------------------------------------
# Datasets
# -------------------------------------------------------

train_dataset = IXIUnpairedDataset(
    lf_dir=dataset_cfg["ixi"]["lf"],
    hf_dir=dataset_cfg["ixi"]["hf"],
    transform=train_transform,
)

val_dataset = PMCPairedDataset(
    lf_dir=dataset_cfg["pmc"]["lf"],
    hf_dir=dataset_cfg["pmc"]["hf"],
    split_file="splits/pmc_val.txt",
    transform=val_transform,
)

# -------------------------------------------------------
# DataLoaders
# -------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=train_cfg["batch_size"],
    shuffle=True,
    num_workers=train_cfg["num_workers"],
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=train_cfg["num_workers"],
    pin_memory=True,
)

# -------------------------------------------------------
# Model
# -------------------------------------------------------

model = UNet3D(
    in_channels=model_cfg["input_channels"],
    out_channels=model_cfg["output_channels"],
    base_channels=model_cfg["base_channels"],
)

# -------------------------------------------------------
# Loss
# -------------------------------------------------------

criterion = ReconstructionLoss()

# -------------------------------------------------------
# Optimizer
# -------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_cfg["learning_rate"],
    weight_decay=train_cfg["weight_decay"],
)

# -------------------------------------------------------
# Scheduler
# -------------------------------------------------------

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=train_cfg["epochs"],
)

# -------------------------------------------------------
# Trainer
# -------------------------------------------------------

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    config=train_cfg,
)

# -------------------------------------------------------
# Start Training
# -------------------------------------------------------

trainer.train()