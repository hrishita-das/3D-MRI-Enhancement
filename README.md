# 3D MRI Enhancement using 3D U-Net

## Overview

This repository implements a baseline 3D U-Net for MRI enhancement.

The network learns to translate low-field (1.5T) MRI volumes into high-field (3T) MRI volumes.

Training is performed using:

- Unpaired IXI dataset (training)
- Paired PMC dataset (validation)

---

## Project Structure

```
configs/
datasets/
evaluation/
losses/
models/
preprocessing/
scripts/
splits/
trainers/
utils/
visualization/

train.py
infer.py
requirements.txt
```

---

## Installation

```bash
git clone <repo-url>
cd 3Dframework

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Dataset

Place the datasets as follows:

```
data/

    ixi_unpaired/
        1.5T/
        3T/

    pmc_paired/
        ...
```

Dataset paths are configured in

```
configs/dataset.yaml
```

---

## Training

```
python train.py
```

or

```
nohup python train.py > experiments/lower_bound/logs/train.log 2>&1 &
```

---

## Outputs

Training produces

- checkpoints
- TensorBoard logs
- metrics
- generated outputs

inside

```
experiments/
```

---

## Model

Baseline:

- 3D U-Net
- Instance Normalization
- AdamW optimizer
- Cosine Annealing LR
- Mixed Precision Training (AMP)

---

## Author

Hrishita Das