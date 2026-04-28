#!/usr/bin/env python3
"""
HD NCO Array — MNIST-like Synthetic Test (N=256, C=10)
========================================================
بدون numpy. Pure Python + list comprehensions.
"""

import random
import math
import time

# ─── Parameters ──────────────────────────────────────────────────────────────
N = 256
C = 10
N_STEPS = 5
TH = 1 << 30
SIGMA = 0.06
SEED = 42
W_MIN, W_MAX = 0.5, 2.0
THS_MIN, THS_MAX = 0.01, 0.1
N_TRAIN = 100
N_TEST = 100
ALPHA = 0.1


# ─── Helper: matrix operations without numpy ────────────────────────────────
def mat_mul(A, B):
    """A (m×k) × B (k×n) → C (m×n)"""
    m, k1 = len(A), len(A[0])
    k2, n = len(B), len(B[0])
    assert k1 == k2, f"mat_mul dim mismatch: {k1} vs {k2}"
    return [[sum(A[i][k] * B[k][j] for k in range(k1)) for j in range(n)] for i in range(m)]

def mat_vec_mul(M, v):
    """M (m×n) × v (n) → result (m)"""
    m, n = len(M), len(M[0])
    return [sum(M[i][k] * v[k] for k in range(n)) for i in range(m)]

def transpose(M):
    """M (m×n) → M^T (n×m)"""
    m, n = len(M), len(M[0])
    return [[M[i][j] for i in range(m)] for j in range(n)]

def vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def vec_scale(v, s):
    return [x * s for x in v]

def vec_mean(vectors):
    """Mean of list of vectors (same length)."""
    n = len(vectors)
    if n == 0: return []
    dim = len(vectors[0])
    return [sum(vec[d] for vec in vectors) / n for d in range(dim)]

def mat_identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


# ─── Ridge regression: (X^T X + alpha I)^{-1} X^T Y ────────────────────────
def ridge_regression(X, Y, alpha):
    """
    X: list of N_samples × N
    Y: list of N_samples × C
    Returns W: C × N
    Uses Gaussian elimination (no numpy).
    """
    N_samples = len(X)
    N_feat = len(X[0])
    C_out = len(Y[0])
    
    # Compute X^T X (N_feat × N_feat)
    XT = transpose(X)
    XTX = mat_mul(XT, X)  # N x N
    
    # Add alpha * I
    for i in range(N_feat):
        XTX[i][i] += alpha
    
    # Compute X^T Y (N_feat × C_out)
    XTY = mat_mul(XT, Y)  # N x C
    
    # Solve for each column of W
    # W_col = (X^T X)^{-1} * XTY_col
    W_cols = []
    for c in range(C_out):
        b = [XTY[i][c] for i in range(N_feat)]
        # Gaussian elimination with partial pivoting
        A = [row[:] for row in XTX]  # copy
        
        # Forward elimination
        for col in range(N_feat):
            # Find pivot
            max_row = max(range(col, N_feat), key=lambda r: abs(A[r][col]))
            if abs(A[max_row][col]) < 1e-15:
                continue
            # Swap
            A[col], A[max_row] = A[max_row], A[col]
            b[col], b[max_row] = b[max_row], b[col]
            pivot = A[col][col]
            # Normalize row
            for j in range(col, N_feat):
                A[col][j] /= pivot
            b[col] /= pivot
            # Eliminate below
            for r in range(col + 1, N_feat):
                factor = A[r][col]
                if abs(factor) < 1e-15:
                    continue
                for j in range(col, N_feat):
                    A[r][j] -= factor * A[col][j]
                b[r] -= factor * b[col]
        
        # Back substitution
        x = [0.0] * N_feat
        for i in range(N_feat - 1, -1, -1):
            s = b[i]
            for j in range(i + 1, N_feat):
                s -= A[i][j] * x[j]
            if abs(A[i][i]) > 1e-15:
                x[i] = s / A[i][i]
            else:
                x[i] = 0.0
        W_cols.append(x)
    
    # Transpose: W_cols is C × N, return as list of C lists of length N
    return W_cols


# ─── Generate Synthetic Digits ──────────────────────────────────────────────
def draw_digit_28x28(digit, noise_std=0.0):
    img = [0.0] * 784
    def set_pixel(row, col, val=1.0):
        if 0 <= row < 28 and 0 <= col < 28:
            img[row * 28 + col] = val
    def draw_line(r1, c1, r2, c2, val=1.0):
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        steps = max(dr, dc)
        if steps == 0:
            set_pixel(r1, c1, val)
            return
        for i in range(steps + 1):
            r = int(r1 + (r2 - r1) * i / steps)
            c = int(c1 + (c2 - c1) * i / steps)
            set_pixel(r, c, val)
            set_pixel(r+1, c, val*0.7)
            set_pixel(r, c+1, val*0.7)
            set_pixel(r-1, c, val*0.7)
            set_pixel(r, c-1, val*0.7)
    
    cx, cy = 14, 14
    if digit == 0:
        for a in range(0, 360, 5):
            r = cx + int(8 * math.cos(a * math.pi / 180))
            c = cy + int(10 * math.sin(a * math.pi / 180))
            set_pixel(r, c, 1.0)
    elif digit == 1:
        draw_line(4, 14, 6, 10, 0.8)
        draw_line(6, 10, 24, 14, 1.0)
    elif digit == 2:
        draw_line(4, 8, 4, 18, 0.8)
        draw_line(4, 18, 12, 18, 0.8)
        draw_line(12, 18, 12, 14, 0.8)
        draw_line(12, 14, 4, 6, 0.8)
        draw_line(4, 6, 14, 6, 0.8)
    elif digit == 3:
        draw_line(6, 10, 6, 18, 1.0)
        draw_line(6, 10, 14, 10, 0.8)
        draw_line(6, 14, 14, 14, 0.8)
        draw_line(6, 18, 22, 18, 0.8)
        draw_line(22, 18, 22, 10, 0.8)
    elif digit == 4:
        draw_line(10, 6, 10, 20, 1.0)
        draw_line(10, 14, 22, 14, 1.0)
        draw_line(20, 6, 20, 22, 0.8)
    elif digit == 5:
        draw_line(18, 8, 6, 8, 0.8)
        draw_line(6, 8, 6, 14, 1.0)
        draw_line(6, 14, 18, 14, 0.8)
        draw_line(18, 14, 18, 22, 1.0)
        draw_line(18, 22, 6, 22, 0.8)
    elif digit == 6:
        draw_line(20, 8, 8, 8, 0.8)
        draw_line(8, 8, 8, 22, 1.0)
        draw_line(8, 22, 20, 22, 0.8)
        draw_line(20, 22, 20, 16, 0.8)
        draw_line(20, 16, 12, 16, 0.8)
    elif digit == 7:
        draw_line(6, 8, 22, 8, 1.0)
        draw_line(22, 8, 16, 22, 0.8)
    elif digit == 8:
        draw_line(8, 8, 8, 22, 1.0)
        draw_line(20, 8, 20, 22, 1.0)
        draw_line(8, 8, 20, 8, 0.8)
        draw_line(8, 22, 20, 22, 0.8)
        draw_line(8, 15, 20, 15, 0.8)
    elif digit == 9:
        draw_line(8, 8, 20, 8, 0.8)
        draw_line(20, 8, 20, 22, 1.0)
        draw_line(8, 8, 8, 14, 0.8)
        draw_line(8, 14, 20, 14, 0.8)
    if noise_std > 0:
        rng = random.Random(SEED + digit * 1000)
        for i in range(784):
            img[i] = max(0, min(1, img[i] + rng.gauss(0, noise_std)))
    return img


print("Generating synthetic digits...")
rng = random.Random(SEED)
base_images = {d: draw_digit_28x28(d, noise_std=0.0) for d in range(10)}

X_train, y_train = [], []
for c in range(C):
    for _ in range(N_TRAIN):
        r = random.Random(SEED + c * 10000 + _)
        noise = [r.gauss(0, 0.15) for _ in range(784)]
        img = [max(0, min(1, base_images[c][i] + noise[i])) for i in range(784)]
        X_train.append(img)
        y_train.append(c)

X_test, y_test = [], []
for c in range(C):
    for _ in range(N_TEST):
        r = random.Random(SEED + c * 100000 + _)
        sx, sy = r.randint(-2, 2), r.randint(-2, 2)
        shifted = [0.0] * 784
        base = base_images[c]
        for row in range(28):
            for col in range(28):
                sr = (row + sy) % 28
                sc = (col + sx) % 28
                shifted[row * 28 + col] = base[sr * 28 + sc]
        noise = [r.gauss(0, 0.15) for _ in range(784)]
        img = [max(0, min(1, shifted[i] + noise[i])) for i in range(784)]
        X_test.append(img)
        y_test.append(c)

for i in range(len(X_test)):
    r = random.Random(SEED + 1000000 + i)
    bright = 1.0 + r.uniform(-0.2, 0.2)
    X_test[i] = [max(0, min(1, v * bright)) for v in X_test[i]]

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# ─── Initialize NCO Array ────────────────────────────────────────────────────
print(f"\nNCO Array: N={N}")
nco_weights = [rng.uniform(W_MIN, W_MAX) for _ in range(N)]
nco_ths = [rng.uniform(THS_MIN, THS_MAX) * TH for _ in range(N)]
nco_seeds = [rng.randint(0, 2**31-1) for _ in range(N)]

# Sparse random projections (10% connectivity)
nco_proj = []
for i in range(N):
    r_local = random.Random(nco_seeds[i])
    idxs = sorted(random.sample(range(784), 78))
    wts = [r_local.uniform(-1, 1) for _ in range(78)]
    nco_proj.append((idxs, wts))

def compute_state(pixels):
    state = []
    for i in range(N):
        idxs, wts = nco_proj[i]
        proj = sum(pixels[idx] * wts[j] for j, idx in enumerate(idxs))
        proj = max(0, proj / 78.0)
        f_word = int(proj * nco_weights[i] * TH)
        ths = nco_ths[i]
        r_noise = random.Random(nco_seeds[i] + 1)
        acc, count = 0, 0
        for _ in range(N_STEPS):
            acc += f_word + int(r_noise.gauss(0, SIGMA * TH))
            if acc >= ths:
                count += 1
                acc = 0
        state.append(count)
    return state


# ─── Compute Training States ──────────────────────────────────────────────────
print(f"\nTraining states ({len(X_train)} samples)...")
t0 = time.time()
X_states = []
for i, img in enumerate(X_train):
    X_states.append(compute_state(img))
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(X_train)} ({time.time()-t0:.1f}s)")
print(f"  Done ({time.time()-t0:.1f}s)")

# Stats
all_firings = [s for state in X_states for s in state]
mean_f = sum(all_firings) / len(all_firings)
num_zero = sum(1 for s in all_firings if s == 0)
print(f"\n  Firing stats: mean={mean_f:.3f}, sparsity={num_zero/len(all_firings)*100:.1f}%")

# Per-class mean
print(f"  Per-class firing means:")
for c in range(C):
    idxs = [j for j in range(len(y_train)) if y_train[j] == c]
    means = [sum(X_states[j][i] for j in idxs) / len(idxs) for i in range(N)]
    print(f"    Class {c}: mean={sum(means)/N:.3f}")

# ─── Ridge Regression ────────────────────────────────────────────────────────
print(f"\nTraining (ridge, alpha={ALPHA})...")
Y_onehot = [[1 if j == y_train[i] else 0 for j in range(C)] for i in range(len(y_train))]
W_out = ridge_regression(X_states, Y_onehot, ALPHA)  # C × N
print(f"  W_out: {C}×{N}")

# Training accuracy
correct = 0
for i in range(len(X_states)):
    scores = mat_vec_mul(W_out, X_states[i])
    pred = max(range(C), key=lambda c: scores[c])
    if pred == y_train[i]:
        correct += 1
train_acc = correct / len(X_states) * 100
print(f"  Training accuracy: {train_acc:.2f}%")

# ─── Test ─────────────────────────────────────────────────────────────────────
print(f"\nTest ({len(X_test)} samples)...")
t0 = time.time()
X_test_states = []
for i, img in enumerate(X_test):
    X_test_states.append(compute_state(img))
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(X_test)} ({time.time()-t0:.1f}s)")
print(f"  Done ({time.time()-t0:.1f}s)")

correct = 0
confusion = [[0]*C for _ in range(C)]
for i in range(len(X_test_states)):
    scores = mat_vec_mul(W_out, X_test_states[i])
    pred = max(range(C), key=lambda c: scores[c])
    if pred == y_test[i]:
        correct += 1
    confusion[y_test[i]][pred] += 1

test_acc = correct / len(X_test_states) * 100

# ─── Results ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"HD NCO Array (N={N}, C={C}) — MNIST-like Test")
print("="*60)
print(f"  Training: {train_acc:.2f}%")
print(f"  Test:     {test_acc:.2f}%")
print()
for c in range(C):
    total = sum(confusion[c])
    correct_c = confusion[c][c]
    print(f"  Class {c}: {correct_c}/{total} = {correct_c/total*100:.1f}%")
print()
print("  Confusion matrix (rows=true, cols=pred):")
print("     " + " ".join(f"{p:3d}" for p in range(C)))
for c in range(C):
    print(f"  {c}: " + " ".join(f"{confusion[c][p]:3d}" for p in range(C)))
print("="*60)
