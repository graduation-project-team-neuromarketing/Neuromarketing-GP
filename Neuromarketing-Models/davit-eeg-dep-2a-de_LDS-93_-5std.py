# %%
"""
# TLDR:
* Model: Davit
* PreProcessing: None
* Data: DE features
* Balancing Technique: Weighted random sampler
* Labels: 0 (Neutral) , 1  Negative (Sad), 2  (Fear) , 3 Positive (Happy)
* Test Accuracy:
"""

# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torcheeg import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torcheeg.datasets import SEEDIVFeatureDataset
from DaViT.models.davit import DaViT
from sklearn.model_selection import train_test_split
from torchvision.transforms import RandomErasing
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import time
import os
import shutil
import random
import math

import numpy as np

from torcheeg.datasets.constants import SEED_IV_CHANNEL_LOCATION_DICT

# --- THE MAIN SUBJECT LOOP ---
import numpy as np
import matplotlib.pyplot as plt

# Emotiv EPOC+
target_channels = [
    "AF3",
    "AF4",
    "F7",
    "F8",
    "F3",
    "F4",
    "FC5",
    "FC6",
    "T7",
    "T8",
    "P7",
    "P8",
    "O1",
    "O2",
]


print(f"Emotiv EPOC+: {target_channels}")
all_channels = list(SEED_IV_CHANNEL_LOCATION_DICT.keys())
target_indices = [all_channels.index(ch) for ch in target_channels]


# %%
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

# %%
# used for logging
TRIAL_NUMBER = 53

# %%
USE_GPU = 1
BATCH_SIZE = 64
DROPOUT = 0.5
ATTN_DROPOUT = 0.5
LR = 5e-4
WEIGHT_DECAY = 2e-1
EPOCHS_COUNT = 250
EARLY_STOP = 1
patience = 25

# %%
# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")
print(f"device: {device}")


# %%
def map_emotions(x):
    # Input x is 0, 1, 2, or 3
    if x == 1 or x == 2:
        return 0  # Sad (1) and Fear (2) become Negative -> (0)    elif x == 3:
    elif x == 3:
        return 1  # Happy (3) becomes Positive -> (1)
    return 0


# %%
class RemapLabel:
    def __init__(self, map_func):
        self.map_func = map_func

    def __call__(self, y):
        return {"y": self.map_func(y)}


# %%
io_path = f"./tmp_out/seed_iv_feautres_{TRIAL_NUMBER}"
if os.path.exists(io_path):
    shutil.rmtree(io_path)
root_path = "./SEED-IV/eeg_feature_smooth/"

# io_path = "./tmp_out/seed_iv_features"
# root_path = "/kaggle/input/seed-iv/eeg_feature_smooth"

# %%
# 3. Load Data

dataset = SEEDIVFeatureDataset(
    io_path=io_path,
    root_path=root_path,
    feature=["de_LDS"],
    offline_transform=transforms.Compose(
        [
            # transforms.PickElectrode(target_indices),
            # transforms.BaselineRemoval(),
            transforms.To2d(),
            transforms.ToTensor(),
        ]
    ),
    label_transform=transforms.Compose(
        [
            transforms.Select("emotion"),
            RemapLabel(map_emotions),
            # RemapLabel(map_emotions2),
        ]
    ),
    online_transform=transforms.Compose([transforms.MeanStdNormalize(axis=(1, 2))]),
    io_mode="memory",
)


# %%
counts = dataset.info["emotion"].value_counts().sort_index()
total = len(dataset.info)

print(f"Total Segments (before droping 0): {total}")
print("Count per Emotion:")
print(counts)
print("-" * 30)

# %%
# 1. Get the metadata DataFrame
df = dataset.info
df = df[df["emotion"] != 0]  # drop neutral
# df = dataset.info.groupby(
#     ["subject_id", "session_id", "trial_id"], group_keys=False
# ).apply(lambda x: x.iloc[:30])


# 2. Count the segments for each emotion
# 0: (Sad, Fear), 1: Happy
counts = df["emotion"].value_counts().sort_index()
total = len(df)

print(f"Total Segments: {total}")
print("-" * 30)
print("Count per Emotion:")
print(counts)

print("-" * 30)
print("Percentage per Emotion:")
percentages = (counts / total) * 100
print(percentages.round(2))

# 3. Check for Imbalance
# If the difference between max and min is > 10%, we might need a WeightedSampler
max_pct = percentages.max()
min_pct = percentages.min()

if (max_pct - min_pct) > 10:
    print(f"\n⚠️ WARNING: Data is IMBALANCED (Diff: {max_pct - min_pct:.2f}%)")
    print("Consider using a WeightedRandomSampler.")
else:
    print(f"\n✅ Data is reasonably BALANCED (Diff: {max_pct - min_pct:.2f}%)")

# %%
# 4. Check Data Shape & Setup Model
# We fetch one sample to determine input channels and dimensions
# (using the whole dataset for detection is fine as shape is consistent)
temp_loader = DataLoader(dataset, batch_size=1, shuffle=False)
temp_X, _ = next(iter(temp_loader))
input_channels = temp_X.shape[1]
print(f"Detected Input Shape: {temp_X.shape}")
print(f"Setting model in_chans to: {input_channels}")


# %%
def DaViT_eeg(
    pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs
):
    model_kwargs = dict(
        patch_size=4,
        in_chans=input_channels,
        window_size=3,
        # Restoring "Tiny" dimensions (User has VRAM headroom)
        # embed_dims=(96, 192, 384, 768),
        # num_heads=(3, 6, 12, 24),
        # embed_dims=(64, 128, 256, 512),
        # num_heads=(2, 4, 8, 16),
        # embed_dims=(48, 96, 192, 384),
        # num_heads=(2, 3, 8, 16),
        embed_dims=(32, 64, 128, 256),
        num_heads=(1, 2, 4, 8),
        depths=(1, 1, 1, 1),
        mlp_ratio=4.0,
        overlapped_patch=False,
        num_classes=kwargs.get("_num_classes", 4),
        drop_rate=kwargs.get("dropout", 0.5),
        attn_drop_rate=kwargs.get("attn_dropout", 0.5),
        drop_path_rate=0.3,  # Add Stochastic Depth
        **kwargs,
    )

    return DaViT(**model_kwargs)


# %%
subjects = sorted(df["subject_id"].unique())  # Get list of all 15 subjects


# %%
# --- MIXUP HELPER FUNCTIONS ---
def mixup_data(x, y, alpha=1.0):
    """
    Applies Mixup augmentation to a batch of data.

    Logic:
    1. Generate a mixing coefficient (lambda) from a Beta(alpha, alpha) distribution.
       - If alpha is close to 0, lambda is close to 0 or 1 (little mixing).
       - If alpha is large, lambda is close to 0.5 (strong mixing).
    2. Shuffle the batch indices to get a "partner" for every sample.
    3. Create a new "mixed" input: mixed_x = lambda * x + (1 - lambda) * x[shuffled_indices]

    Returns:
        mixed_x: The blended input tensor.
        y_a: The original labels.
        y_b: The labels of the shuffled samples.
        lam: The mixing coefficient used.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss.

    Since the input was a mix of two samples (A and B) with weight lambda:
    Loss = lambda * Loss(pred, Label_A) + (1 - lambda) * Loss(pred, Label_B)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# %%
train_transforms = transforms.Compose(
    [
        transforms.RandomMask(p=0.1),
        transforms.RandomNoise(p=0.2),
    ]
)

# Initialize Random Erasing
# p=0.5: 50% chance to apply
# scale=(0.02, 0.25): Erase smaller chunks to avoid killing too much info
# ratio=(0.05, 20.0):
#   - 0.05 means aspect ratio 1:20 (Very wide/flat, erasing a specific band across many channels)
#   - 20.0 means aspect ratio 20:1 (Very tall/thin, erasing all bands for a few channels)
# This flexibility suits the 62x5 shape much better than the default (0.3, 3.3).
eraser = RandomErasing(
    p=0.2, scale=(0.02, 0.25), ratio=(0.05, 20.0), value=0, inplace=False
)

all_targets = []
all_predictions = []

# %%
all_subject_accuracies = []
all_subject_histories = {}
for subject_id in subjects:
    print("\n========================================")
    print(f"  PROCESSING SUBJECT: {subject_id}")
    print("========================================")

    # 1. Filter Data for this Subject
    sub_df = df[df["subject_id"] == subject_id]
    # 2. Split by Unique Video (Session + Trial) for this subject
    # CRITICAL: We cannot just split by 'trial_id' because Trial 1 in Session 1 (Sad)
    # is different from Trial 1 in Session 2 (Fear). We must treat them as separate videos.

    # Create a unique ID for every video: "session_trial" (e.g., "1_5", "2_5", "3_5")
    sub_df = sub_df.copy()  # Avoid SettingWithCopy warning
    sub_df["unique_run_id"] = (
        sub_df["session_id"].astype(str) + "_" + sub_df["trial_id"].astype(str)
    )

    # Extract labels per unique run to ensure balanced split
    # We get a single label for each unique run (session_trial)
    # Since every segment in a trial has same label, we just drop duplicates
    run_info = sub_df[["unique_run_id", "emotion"]].drop_duplicates()
    all_runs = run_info["unique_run_id"].values
    all_labels = run_info["emotion"].values

    # Stratified Split at the RUN level
    # This keeps 80/20 ratio and ensures each class is represented in the test set
    train_runs, test_runs = train_test_split(
        all_runs, test_size=0.20, random_state=42, stratify=all_labels
    )

    print(f"Total Unique Videos (Across 3 Sessions): {len(all_runs)}")
    print(f"Training on: {len(train_runs)} | Testing on: {len(test_runs)}")

    # Extract indices (Zero Leakage Guaranteed)
    train_indices = sub_df[sub_df["unique_run_id"].isin(train_runs)].index.tolist()
    test_indices = sub_df[sub_df["unique_run_id"].isin(test_runs)].index.tolist()

    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, test_indices)

    y_train_indices = train_set.indices
    raw_labels = dataset.info.iloc[y_train_indices]["emotion"].values
    mapped_labels = [map_emotions(y) for y in raw_labels]
    # mapped_labels = raw_labels

    # 3. Create Specific Sampler for this Subject
    class_counts = np.bincount(mapped_labels)
    print(f"class_counts: {class_counts}")
    # class_weights = 1.0 / class_counts
    classes_ = np.unique(mapped_labels)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_,
        y=mapped_labels,
    )
    print(f"class_weights: {class_weights}")
    # sample_weights = [class_weights[y] for y in mapped_labels]

    # 3. Create Specific Sampler for this Subject
    # class_counts = np.bincount(raw_labels)
    # class_weights = 1.0 / class_counts
    # print(class_weights)
    # sample_weights = [class_weights[y] for y in raw_labels]

    # sampler = WeightedRandomSampler(
    #     weights=sample_weights, num_samples=len(sample_weights), replacement=True
    # )

    # 4. Loaders
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    train_count = len(train_loader)
    test_count = len(test_loader)

    # 5. FRESH Model & Optimizer (Reset for every subject)
    model = DaViT_eeg(
        dropout=DROPOUT, attn_dropout=ATTN_DROPOUT, _num_classes=len(class_weights)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine Annealing: Smoothly drops LR from 0.0001 -> 0 over EPOCHS_COUNT
    # This forces the model to settle into the best accuracy at the end
    # scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_COUNT, eta_min=1e-6)

    # # OneCycleLR is better for transformers (Warmup + Annealing)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=EPOCHS_COUNT,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # 30% warmup
        div_factor=10,
        final_div_factor=1000,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights).to(device).float(), label_smoothing=0.1
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 6. Training Loop for this Subject
    best_acc = 0.0
    best_tacc = 0.0
    best_loss = float("inf")
    counter = 0

    subject_train_acc = []
    subject_val_acc = []
    prinT = 1

    # --- MAIN LOOP EXTENSION ---
    for epoch in range(EPOCHS_COUNT):  # 40 epochs is usually enough for 1 subject
        startTime = time.time()
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            X, y = batch
            # Augmentation: Random Erasing
            X = train_transforms(eeg=X)["eeg"]
            # X = eraser(X)

            if prinT:
                print(f"X: {X.shape},y: {y.shape}")
                prinT = 0

            X = X.to(device)
            y = y.to(device).long()

            # 2. Apply Mixup
            X, y_a, y_b, lam = mixup_data(X, y, alpha=0.2)
            X, y_a, y_b = map(torch.autograd.Variable, (X, y_a, y_b))

            optimizer.zero_grad()
            outputs = model(X)

            # 3. Use Mixup Loss
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            optimizer.step()

            # For accuracy, we just compare against the dominant label (or y_a)
            # Mixup creates "soft" accuracy, but standard accuracy is fine for logging
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (
                lam * predicted.eq(y_a.data).cpu().sum().float()
                + (1 - lam) * predicted.eq(y_b.data).cpu().sum().float()
            ).item()
            # scheduler step for OneCycleLR is per batch
            scheduler.step()
            train_loss += loss.item()

        train_acc = (correct / total) * 100
        avg_train_loss = train_loss / train_count

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                X, y = batch

                X = X.to(device)
                y = y.to(device).long()

                output = model(X)

                loss = criterion(output, y)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
                all_targets.append(y)
                all_predictions.append(predicted)

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / test_count
        current_lr = optimizer.param_groups[0]["lr"]
        elabsedTime = time.time() - startTime
        loss_ratio = avg_train_loss / avg_val_loss
        subject_train_acc.append(train_acc)
        subject_val_acc.append(val_acc)

        # Early Stopping check (Monitor Validation Loss)
        is_best_acc = False
        if (val_acc > best_acc) or (
            (best_acc - val_acc) <= 0.5 and train_acc >= best_tacc
        ):
            # if avg_val_loss < best_loss:
            is_best_acc = True
            best_acc = val_acc
            best_tacc = train_acc
            best_loss = avg_val_loss
            counter = 0
        else:
            # Monitor Loss for Early Stopping (No Model Saving)
            counter += 1
            if EARLY_STOP and counter >= patience:
                break

        print(
            f"Epoch {epoch + 1}: Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}% | Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | Loss Ratio: {loss_ratio:.2f} | LR: {current_lr:.6f} | Elapsed Time: {elabsedTime:.2f} seconds {'[BEST ACC]' if is_best_acc else ''}"
        )

    all_subject_accuracies.append(best_acc)
    all_subject_histories[subject_id] = {
        "train": subject_train_acc,
        "val": subject_val_acc,
    }

# %%
# --- FINAL RESULTS ---
avg_acc = sum(all_subject_accuracies) / len(all_subject_accuracies)
print("\n========================================")
print(f"FINAL AVERAGE ACCURACY: {avg_acc:.2f}%")
print(f"Detailed: {all_subject_accuracies}")
print("========================================")


# %%
# --- PLOTTING ---
# 1. 5x3 Grid for individual subjects
fig, axes = plt.subplots(5, 3, figsize=(20, 15))
axes = axes.flatten()

for i, subject_id in enumerate(subjects):
    if i < len(axes):
        ax = axes[i]
        if subject_id in all_subject_histories:
            h = all_subject_histories[subject_id]
            ax.plot(h["train"], label="Train Acc")
            ax.plot(h["val"], label="Val Acc")
            ax.set_title(f"Subject {subject_id}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.legend()
            ax.grid(True)

plt.tight_layout()
plt.savefig(f"subject_accuracies {TRIAL_NUMBER}.png")
print("Saved subject_accuracies.png")
# plt.show()

# %%
# 2. Average Plot
plt.figure(figsize=(10, 6))

# Find max epochs across all runs
max_epochs = 0
for h in all_subject_histories.values():
    max_epochs = max(max_epochs, len(h["train"]))

avg_train = []
avg_val = []

for e in range(max_epochs):
    t_vals = []
    v_vals = []
    for h in all_subject_histories.values():
        # Use last value if a subject finished early
        idx = min(e, len(h["train"]) - 1)
        t_vals.append(h["train"][idx])
        v_vals.append(h["val"][idx])

    avg_train.append(np.mean(t_vals))
    avg_val.append(np.mean(v_vals))

plt.plot(avg_train, label="Average Train Acc", linewidth=2)
plt.plot(avg_val, label="Average Val Acc", linewidth=2)
plt.title("Average Accuracy Across All Subjects")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.savefig(f"average_accuracy {TRIAL_NUMBER}.png")
print("Saved average_accuracy.png")
# plt.show()

# %%
# Concatenate all stored tensors
all_targets = torch.cat(all_targets)
all_predictions = torch.cat(all_predictions)
y_true = all_targets.view(-1).cpu().numpy()
y_pred = all_predictions.view(-1).cpu().numpy()
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap=plt.cm.Blues)
plt.savefig(f"validation cm {TRIAL_NUMBER}.png")
