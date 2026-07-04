import os
import glob
import random

random.seed(42)

lf_dir = "data/pmc_paired/PMC dataset/3D/created-3D/1o5T_final/T1_1o5T"

files = sorted(glob.glob(os.path.join(lf_dir, "*.nii.gz")))

patient_ids = [
    os.path.basename(f).replace(".nii.gz", "")
    for f in files
]

random.shuffle(patient_ids)

val_size = int(0.70 * len(patient_ids))

val_ids = patient_ids[:val_size]

test_ids = patient_ids[val_size:]

os.makedirs("splits", exist_ok=True)

with open("splits/pmc_val.txt", "w") as f:
    for pid in val_ids:
        f.write(pid + "\n")

with open("splits/pmc_test.txt", "w") as f:
    for pid in test_ids:
        f.write(pid + "\n")

print("Total:", len(patient_ids))
print("Validation:", len(val_ids))
print("Test:", len(test_ids))