#!/usr/bin/env python3
"""
HD NCO Array — MNIST Real v3 (N=256, C=10)
=============================================
نسخة محسّنة بـ numpy. تشغيل على MNIST الحقيقي
في GitHub Codespaces.

Parameters: identical to v2 for fairness comparison.
"""

import numpy as np
import struct
import gzip
import os
import time
import random

# ─── Parameters ──────────────────────────────────────────────────────────────
N = 256
C = 10
N_STEPS = 5
TH = 1 << 30
SIGMA = 0.06
SEED = 42
W_MIN, W_MAX = 0.5, 2.0
THS_MIN, THS_MAX = 0.01, 0.1
ALPHA = 5.0

# ─── Load MNIST ──────────────────────────────────────────────────────────────
def load_mnist(images_path, labels_path, limit=None):
    with gzip.open(images_path, 'rb') as f:
        f.read(16)
        data = f.read()
    if limit:
        n_images = limit
        data = data[:limit * 784]
    else:
        n_images = len(data) // 784
    images = np.frombuffer(data, dtype=np.uint8).reshape(n_images, 784).astype(np.float32) / 255.0
    
    with gzip.open(labels_path, 'rb') as f:
        f.read(8)
        labels = np.frombuffer(f.read(n_images), dtype=np.uint8)
    return images, labels

def load_mnist_binary(images_path, labels_path):
    """Load raw ubyte (already decompressed) files."""
    with open(images_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(num * rows * cols), dtype=np.uint8)
        images = data.reshape(num, rows * cols).astype(np.float32) / 255.0
    with open(labels_path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(num), dtype=np.uint8)
    return images, labels

print("Loading MNIST...")
data_dir = "."
# Files are in the repo root (pushed from Termux)

# Try .gz files in current directory (preferred — already exist)
train_im_gz = "train-images-idx3-ubyte.gz"
train_lb_gz = "train-labels-idx1-ubyte.gz"
test_im_gz = "t10k-images-idx3-ubyte.gz"
test_lb_gz = "t10k-labels-idx1-ubyte.gz"

train_raw = "train-images-idx3-ubyte"
train_lb_raw = "train-labels-idx1-ubyte"
test_raw = "t10k-images-idx3-ubyte"
test_lb_raw = "t10k-labels-idx1-ubyte"

if all(os.path.exists(f) for f in [train_raw, train_lb_raw, test_raw, test_lb_raw]):
    train_im, train_lb = load_mnist_binary(train_raw, train_lb_raw)
    test_im, test_lb = load_mnist_binary(test_raw, test_lb_raw)
elif all(os.path.exists(f) for f in [train_im_gz, train_lb_gz, test_im_gz, test_lb_gz]):
    train_im, train_lb = load_mnist(train_im_gz, train_lb_gz)
    test_im, test_lb = load_mnist(test_im_gz, test_lb_gz)
else:
    # Try to download
    import urllib.request
    base = "https://yann.lecun.com/exdb/mnist/"
    files = [
        ("train-images-idx3-ubyte.gz", train_im_gz),
        ("train-labels-idx1-ubyte.gz", train_lb_gz),
        ("t10k-images-idx3-ubyte.gz", test_im_gz),
        ("t10k-labels-idx1-ubyte.gz", test_lb_gz),
    ]
    for fname, path in files:
        if not os.path.exists(path):
            print(f"  Downloading {fname}...")
            urllib.request.urlretrieve(base + fname, path)
    train_im, train_lb = load_mnist(train_im_gz, train_lb_gz)
    test_im, test_lb = load_mnist(test_im_gz, test_lb_gz)

print(f"  Train: {train_im.shape[0]} samples")
print(f"  Test:  {test_im.shape[0]} samples")

# Balance training set: 500 per class (same as v2)
N_TRAIN_PER_CLASS = 500
print(f"\nBalancing training: {N_TRAIN_PER_CLASS} per class...")
X_train_list = []
y_train_list = []
for c in range(C):
    idxs = np.where(train_lb == c)[0][:N_TRAIN_PER_CLASS]
    X_train_list.append(train_im[idxs])
    y_train_list.append(np.full(N_TRAIN_PER_CLASS, c, dtype=np.uint8))

X_train = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)
print(f"  Training set: {len(X_train)} samples")

# Use all test set
N_TEST = len(test_im)
print(f"  Test set: {N_TEST} samples")

# ─── NCO Array ────────────────────────────────────────────────────────────────
rng = np.random.RandomState(SEED)
print(f"\nInitializing NCO Array: N={N}...")

nco_weights = rng.uniform(W_MIN, W_MAX, N).astype(np.float64)
nco_ths = rng.uniform(THS_MIN, THS_MAX, N).astype(np.float64) * TH
nco_seeds = rng.randint(0, 2**31-1, N)

# Sparse bipolar projections (10% of 784 = 78 per NCO)
print("  Generating bipolar projections...")
nco_indices = np.array([np.sort(rng.choice(784, 78, replace=False)) for _ in range(N)])
nco_bipolar = np.where(rng.rand(N, 78) < 0.5, 1.0, -1.0)


def compute_states_batch(images):
    """Compute NCO states for a batch of images using numpy vectorization.
    
    For each image: project onto NCOs, integrate with noise.
    Returns: (n_samples, N) firing counts.
    """
    n_samples = len(images)
    states = np.zeros((n_samples, N), dtype=np.int32)
    
    # For each NCO, compute projection for all samples
    for i in range(N):
        idxs = nco_indices[i]
        wts = nco_bipolar[i]
        
        # Project: sum(pixels[idx] * wts) for each sample
        proj = images[:, idxs] @ wts  # (n_samples,) dot product
        proj = np.maximum(0, proj / 78.0)  # rectify + normalize
        
        # NCO integration
        f_word = (proj * nco_weights[i] * TH).astype(np.int64)
        ths = int(nco_ths[i])
        
        # Noise per NCO per sample is fixed (same seed per NCO)
        local_rng = np.random.RandomState(int(nco_seeds[i]) + 1)
        noise = local_rng.randn(n_samples, N_STEPS).astype(np.float64) * SIGMA * TH
        noise_int = noise.astype(np.int64)
        
        # Integrate over N_STEPS
        acc = np.zeros(n_samples, dtype=np.int64)
        counts = np.zeros(n_samples, dtype=np.int32)
        
        for step in range(N_STEPS):
            acc += f_word + noise_int[:, step]
            fired = acc >= ths
            counts += fired
            acc[fired] = 0  # reset after firing
        
        states[:, i] = counts
    
    return states


# ─── Training ─────────────────────────────────────────────────────────────────
print(f"\nComputing training states ({len(X_train)} samples)...")
t0 = time.time()
X_states = compute_states_batch(X_train)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s ({len(X_train)/elapsed:.0f} samples/s)")

# Stats
mean_f = np.mean(X_states)
sparsity = np.mean(X_states == 0) * 100
print(f"  Sparsity: {sparsity:.1f}%, Mean firing: {mean_f:.3f}")

# Per-class means
for c in range(C):
    mask = y_train == c
    means = X_states[mask].mean(axis=0)
    print(f"    Class {c}: {means.mean():.3f}")

# Ridge regression with numpy
print(f"\nTraining (ridge, alpha={ALPHA})...")
t0 = time.time()
XTX = X_states.T @ X_states
np.fill_diagonal(XTX, XTX.diagonal() + ALPHA * N_TRAIN_PER_CLASS * C)
XTY = X_states.T @ np.eye(C)[y_train]
W_out = np.linalg.solve(XTX, XTY).T  # C x N
print(f"  W_out: {W_out.shape} ({time.time()-t0:.1f}s)")

# Training accuracy
train_preds = X_states @ W_out.T
train_acc = np.mean(np.argmax(train_preds, axis=1) == y_train) * 100
print(f"  Training accuracy: {train_acc:.2f}%")

# ─── Test ─────────────────────────────────────────────────────────────────────
print(f"\nComputing test states ({len(test_im)} samples)...")
t0 = time.time()
X_test_states = compute_states_batch(test_im)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s ({len(test_im)/elapsed:.0f} samples/s)")

test_preds = X_test_states @ W_out.T
test_acc = np.mean(np.argmax(test_preds, axis=1) == test_lb) * 100

# ─── Results ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"HD NCO Array (N={N}, C={C}) — REAL MNIST (numpy)")
print("="*60)
print(f"  Training: {N_TRAIN_PER_CLASS}/class = {len(X_train)} total")
print(f"  Test:     {N_TEST} samples")
print(f"  Sparsity: {sparsity:.1f}%, Mean firing: {mean_f:.3f}")
print(f"  Training accuracy: {train_acc:.2f}%")
print(f"  Test accuracy:     {test_acc:.2f}%")
print()

# Per-class accuracy
for c in range(C):
    mask = test_lb == c
    class_acc = np.mean(np.argmax(test_preds[mask], axis=1) == c) * 100
    print(f"  Class {c}: {class_acc:.1f}% ({mask.sum()} samples)")

# Confusion matrix
print("\n  Confusion matrix:")
confusion = np.zeros((C, C), dtype=int)
for i in range(len(test_lb)):
    confusion[test_lb[i], np.argmax(test_preds[i])] += 1
for c in range(C):
    row = " ".join(f"{confusion[c,p]:4d}" for p in range(C))
    print(f"  {c}: {row}")

print("="*60)

# Compare with synthetic results
print(f"\nComparison: Synthetic v2 = 86.3% | Real MNIST = {test_acc:.1f}%")
print(f"Difference: {test_acc - 86.3:+.1f} percentage points")
