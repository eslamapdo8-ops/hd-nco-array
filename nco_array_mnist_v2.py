#!/usr/bin/env python3
"""
HD NCO Array — MNIST-like Synthetic Test v2 (N=256, C=10)
===========================================================
تحسينات الخبير:
1. random_projection بقيم مستقطبة (±1) بدلاً من uniform
2. ضوضاء غاوسي σ=0.1 على البكسلات قبل الإسقاط
3. دفعة تدريب 5,000 عينة، اختبار 1,000
4. معامل تنظيم alpha=5.0
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
N_TRAIN = 5000  # 500/class
N_TEST = 1000   # 100/class
ALPHA = 5.0     # stronger regularization


# ─── Matrix helpers (pure Python) ────────────────────────────────────────────
def mat_mul(A, B):
    m, k1 = len(A), len(A[0])
    k2, n = len(B), len(B[0])
    assert k1 == k2
    return [[sum(A[i][k] * B[k][j] for k in range(k1)) for j in range(n)] for i in range(m)]

def mat_vec_mul(M, v):
    return [sum(M[i][k] * v[k] for k in range(len(M[0]))) for i in range(len(M))]

def transpose(M):
    m, n = len(M), len(M[0])
    return [[M[i][j] for i in range(m)] for j in range(n)]

def mat_identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def ridge_regression(X, Y, alpha):
    """W (C×N) = (X^T X + alpha I)^{-1} X^T Y"""
    N_samples = len(X)
    N_feat = len(X[0])
    C_out = len(Y[0])
    
    XT = transpose(X)
    XTX = mat_mul(XT, X)
    for i in range(N_feat):
        XTX[i][i] += alpha
    XTY = mat_mul(XT, Y)
    
    W_cols = []
    for c in range(C_out):
        b = [XTY[i][c] for i in range(N_feat)]
        A = [row[:] for row in XTX]
        
        for col in range(N_feat):
            max_row = max(range(col, N_feat), key=lambda r: abs(A[r][col]))
            if abs(A[max_row][col]) < 1e-15:
                continue
            A[col], A[max_row] = A[max_row], A[col]
            b[col], b[max_row] = b[max_row], b[col]
            pivot = A[col][col]
            for j in range(col, N_feat):
                A[col][j] /= pivot
            b[col] /= pivot
            for r in range(col + 1, N_feat):
                factor = A[r][col]
                if abs(factor) < 1e-15:
                    continue
                for j in range(col, N_feat):
                    A[r][j] -= factor * A[col][j]
                b[r] -= factor * b[col]
        
        x = [0.0] * N_feat
        for i in range(N_feat - 1, -1, -1):
            s = b[i]
            for j in range(i + 1, N_feat):
                s -= A[i][j] * x[j]
            x[i] = s / A[i][i] if abs(A[i][i]) > 1e-15 else 0.0
        W_cols.append(x)
    return W_cols


# ─── Generate Realistic Digits ──────────────────────────────────────────────
def draw_digit_28x28(digit, noise_std=0.0):
    """Draw digit with grayscale levels (0.0–1.0) and rounded edges."""
    img = [0.0] * 784
    
    # Star-shaped gaussian blobs for each digit
    cx, cy = 14, 14
    
    if digit == 0:
        # Ring
        for a in range(0, 360, 2):
            r = cx + int(8 * math.cos(a * math.pi / 180))
            c = cy + int(10 * math.sin(a * math.pi / 180))
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    rr, cc = r+dr, c+dc
                    if 0 <= rr < 28 and 0 <= cc < 28:
                        dist = math.sqrt(dr*dr + dc*dc)
                        img[rr*28+cc] = max(img[rr*28+cc], 1.0 - dist*0.3)
                        
    elif digit == 1:
        for row in range(4, 24):
            for dc in range(-1, 2):
                r, c = row, 14 + dc
                if 0 <= r < 28 and 0 <= c < 28:
                    img[r*28+c] = 1.0 - abs(dc)*0.3
        # Top hook
        for c in range(10, 18):
            dr = max(c, 18-c)
            for dd in range(-1, 2):
                r, cc = 5 + dr, c + dd
                if 0 <= r < 28 and 0 <= cc < 28:
                    img[r*28+cc] = 1.0 - abs(dd)*0.3
                    
    elif digit == 2:
        points = [(4,8),(4,18),(12,18),(12,14),(4,6),(14,6)]
        for i in range(len(points)-1):
            r1,c1 = points[i]; r2,c2 = points[i+1]
            dr, dc = r2-r1, c2-c1
            steps = max(abs(dr), abs(dc))
            for s in range(steps+1):
                r = int(r1 + dr*s/steps); c = int(c1 + dc*s/steps)
                for dd in range(-1,2):
                    rr, cc = r+dd//2, c+dd
                    if 0<=rr<28 and 0<=cc<28:
                        img[rr*28+cc] = max(img[rr*28+cc], 1.0-abs(dd)*0.3)
                        
    elif digit == 3:
        for r in range(6,22):
            for c in [10, 14, 18]:
                if c==10 and r>14: continue
                if c==18 and r>20: continue
                for dd in range(-1,2):
                    if 0<=r<28 and 0<=c+dd<28:
                        img[r*28+c+dd] = max(img[r*28+c+dd], 1.0-abs(dd)*0.3)
        # horizontal segments
        for c in range(10, 19):
            for r in [6, 14, 20]:
                for dd in range(-1,2):
                    if 0<=r+dd<28 and 0<=c<28:
                        img[(r+dd)*28+c] = max(img[(r+dd)*28+c], 1.0-abs(dd)*0.3)
                        
    elif digit == 4:
        for r in range(6, 22):
            for c in [10, 20]:
                for dd in range(-1, 2):
                    if 0<=r<28 and 0<=c+dd<28:
                        img[r*28+c+dd] = max(img[r*28+c+dd], 1.0-abs(dd)*0.3)
        for c in range(10, 21):
            for dd in range(-1, 2):
                if 0<=14+dd<28 and 0<=c<28:
                    img[(14+dd)*28+c] = max(img[(14+dd)*28+c], 1.0-abs(dd)*0.3)
                    
    elif digit == 5:
        for c in range(8, 19):
            for r in [6, 14, 22]:
                for dd in range(-1,2):
                    if 0<=r+dd<28 and 0<=c<28:
                        img[(r+dd)*28+c] = max(img[(r+dd)*28+c], 1.0-abs(dd)*0.3)
        for r in range(6, 15):
            for dd in range(-1,2):
                if 0<=r<28 and 0<=8+dd<28:
                    img[r*28+8+dd] = max(img[r*28+8+dd], 1.0-abs(dd)*0.3)
        for r in range(14, 23):
            for dd in range(-1,2):
                if 0<=r<28 and 0<=18+dd<28:
                    img[r*28+18+dd] = max(img[r*28+18+dd], 1.0-abs(dd)*0.3)
                    
    elif digit == 6:
        for r in range(6, 23):
            for c in [8, 18]:
                if c==8 and r<10: continue
                if c==18 and r>16: continue
                for dd in range(-1,2):
                    if 0<=r<28 and 0<=c+dd<28:
                        img[r*28+c+dd] = max(img[r*28+c+dd], 1.0-abs(dd)*0.3)
        for c in range(8, 19):
            for r in [6, 16, 22]:
                for dd in range(-1,2):
                    if 0<=r+dd<28 and 0<=c<28:
                        img[(r+dd)*28+c] = max(img[(r+dd)*28+c], 1.0-abs(dd)*0.3)
                        
    elif digit == 7:
        for c in range(8, 22):
            for dd in range(-1,2):
                if 0<=6+dd<28 and 0<=c<28:
                    img[(6+dd)*28+c] = max(img[(6+dd)*28+c], 1.0-abs(dd)*0.3)
        for r in range(6, 22):
            for dd in range(-1,2):
                if 0<=r<28 and 0<=20+dd<28:
                    img[r*28+20+dd] = max(img[r*28+20+dd], 1.0-abs(dd)*0.3)
                    
    elif digit == 8:
        for c in [8, 15, 18]:
            for r in range(6, 22):
                if c==18 and r<10: continue
                for dd in range(-1,2):
                    if 0<=r<28 and 0<=c+dd<28:
                        img[r*28+c+dd] = max(img[r*28+c+dd], 1.0-abs(dd)*0.3)
        for c in range(8, 16):
            for r in [6, 14, 22]:
                for dd in range(-1,2):
                    if 0<=r+dd<28 and 0<=c<28:
                        img[(r+dd)*28+c] = max(img[(r+dd)*28+c], 1.0-abs(dd)*0.3)
                        
    elif digit == 9:
        for c in range(8, 19):
            for r in [6, 14]:
                for dd in range(-1,2):
                    if 0<=r+dd<28 and 0<=c<28:
                        img[(r+dd)*28+c] = max(img[(r+dd)*28+c], 1.0-abs(dd)*0.3)
        for r in range(6, 22):
            for c in [8, 18]:
                if c==8 and r<12: continue
                for dd in range(-1,2):
                    if 0<=r<28 and 0<=c+dd<28:
                        img[r*28+c+dd] = max(img[r*28+c+dd], 1.0-abs(dd)*0.3)
    
    if noise_std > 0:
        rng = random.Random(SEED + digit * 1000)
        for i in range(784):
            img[i] = max(0, min(1, img[i] + rng.gauss(0, noise_std)))
    
    return img


print("Generating synthetic MNIST v2...")
rng = random.Random(SEED)

# Base clean images
base_images = {d: draw_digit_28x28(d, noise_std=0.0) for d in range(10)}

# ─── Training set (5,000 samples) ────────────────────────────────────────────
# التحسين 3: دفعة تدريب أكبر
print(f"  Training: {N_TRAIN} samples ({N_TRAIN//C}/class)...")
X_train, y_train = [], []
for c in range(C):
    for _ in range(N_TRAIN // C):
        r = random.Random(SEED + c * 10000 + _)
        
        # التحسين 2: ضوضاء غاوسي σ=0.1 على البكسلات
        noise = [r.gauss(0, 0.1) for _ in range(784)]
        
        # Add variability: slight scaling, shift
        sx, sy = r.randint(-1, 1), r.randint(-1, 1)
        base = base_images[c]
        img = [0.0] * 784
        for row in range(28):
            for col in range(28):
                sr, sc = (row+sy)%28, (col+sx)%28
                img[row*28+col] = base[sr*28+sc]
        
        # Add noise + clip
        final = [max(0, min(1, img[i] + noise[i])) for i in range(784)]
        
        # التحسين 1: grayscale variation not just 0/1 — already grayscale
        X_train.append(final)
        y_train.append(c)

# ─── Test set (1,000 samples) ────────────────────────────────────────────────
print(f"  Test: {N_TEST} samples...")
X_test, y_test = [], []
for c in range(C):
    for _ in range(N_TEST // C):
        r = random.Random(SEED + c * 100000 + _)
        sx, sy = r.randint(-2, 2), r.randint(-2, 2)
        base = base_images[c]
        shifted = [0.0] * 784
        for row in range(28):
            for col in range(28):
                sr, sc = (row+sy)%28, (col+sx)%28
                shifted[row*28+col] = base[sr*28+sc]
        
        noise = [r.gauss(0, 0.1) for _ in range(784)]
        bright = 1.0 + r.uniform(-0.15, 0.15)
        final = [max(0, min(1, (shifted[i] + noise[i]) * bright)) for i in range(784)]
        X_test.append(final)
        y_test.append(c)

print(f"  Done: Train={len(X_train)}, Test={len(X_test)}")

# ─── Initialize NCO Array ────────────────────────────────────────────────────
print(f"\nNCO Array: N={N}")
nco_weights = [rng.uniform(W_MIN, W_MAX) for _ in range(N)]
nco_ths = [rng.uniform(THS_MIN, THS_MAX) * TH for _ in range(N)]
nco_seeds = [rng.randint(0, 2**31-1) for _ in range(N)]

# التحسين 1: random_projection بقيم مستقطبة (±1)
print("  Generating bipolar random projections (±1, 10% connectivity)...")
nco_proj = []
for i in range(N):
    r_local = random.Random(nco_seeds[i])
    idxs = sorted(random.sample(range(784), 78))
    wts = [1 if r_local.random() < 0.5 else -1 for _ in range(78)]
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


# ─── Training ─────────────────────────────────────────────────────────────────
print(f"\nComputing training states ({len(X_train)} samples)...")
t0 = time.time()
X_states = []
for i, img in enumerate(X_train):
    X_states.append(compute_state(img))
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{len(X_train)} ({time.time()-t0:.1f}s)")
print(f"  Done ({time.time()-t0:.1f}s)")

# Stats
all_firings = [s for state in X_states for s in state]
mean_f = sum(all_firings) / len(all_firings)
num_zero = sum(1 for s in all_firings if s == 0)
sparsity = num_zero / len(all_firings) * 100
print(f"\n  Sparsity: {sparsity:.1f}%, Mean firing: {mean_f:.3f}")

# Per-class means
print(f"  Per-class firing means:")
for c in range(C):
    idxs = [j for j in range(len(y_train)) if y_train[j] == c]
    means = [sum(X_states[j][i] for j in idxs) / len(idxs) for i in range(N)]
    print(f"    Class {c}: {sum(means)/N:.3f}")

# Ridge regression
print(f"\nTraining (ridge, alpha={ALPHA})...")
Y_onehot = [[1 if j == y_train[i] else 0 for j in range(C)] for i in range(len(y_train))]
t0 = time.time()
W_out = ridge_regression(X_states, Y_onehot, ALPHA)
print(f"  W_out: {C}×{N} ({time.time()-t0:.1f}s)")

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
print(f"HD NCO Array (N={N}, C={C}) — MNIST-like v2")
print("="*60)
print(f"  Parameters: N_TRAIN={N_TRAIN}, N_TEST={N_TEST}, alpha={ALPHA}")
print(f"  Projection: bipolar (±1), 10% connectivity")
print(f"  Pixel noise: σ=0.1 (Gaussian)")
print(f"  Sparsity: {sparsity:.1f}%, Mean firing: {mean_f:.3f}")
print(f"  Simulation time: ~{time.time()-t0:.0f}s (test only)")
print(f"")
print(f"  Training accuracy: {train_acc:.2f}%")
print(f"  Test accuracy:     {test_acc:.2f}%")
print()
for c in range(C):
    total = sum(confusion[c])
    correct_c = confusion[c][c]
    print(f"  Class {c}: {correct_c}/{total} = {correct_c/total*100:.1f}%")
print()
print("  Confusion matrix (rows=true, cols=pred):")
print("     " + " ".join(f"{p:2d}" for p in range(C)))
for c in range(C):
    print(f"  {c}: " + " ".join(f"{confusion[c][p]:2d}" for p in range(C)))
print("="*60)
