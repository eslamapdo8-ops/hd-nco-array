#!/usr/bin/env python3
"""
HD NCO Array — MNIST Prototype
===============================
N=256 NCOs, C=10 classes (MNIST), N_STEPS=5.
Linear Readout via Ridge Regression.

Pipeline:
  1. Load MNIST (28x28 → 784-D vector, normalized [0,1])
  2. For each image: NCO Array → fingerprint vector (length N)
  3. Train Ridge Readout on fingerprints
  4. Test on 10,000 test images
"""

import random
import json
import time

# ========== Constants ==========
N = 256                  # Number of NCOs
C = 10                   # Classes (0-9)
N_STEPS = 5              # Clock cycles per classification
TH = 2**30               # NCO threshold
OFFSET = TH // 2         # Reset offset
ALPHA = 1.0              # Ridge regularization strength
TRAIN_SIZE = 10000       # Use subset of 60k for speed
TEST_SIZE = 5000         # Use subset of 10k for speed

random.seed(42)

# ========== NCO (identical to Onyx V2) ==========
class NCO:
    __slots__ = ('th', 'offset', 'acc', 'fires', 'seed', '_rng')
    def __init__(self, th, offset, seed):
        self.th = th
        self.offset = offset
        self.acc = 0
        self.fires = 0
        self.seed = seed
        self._rng = random.Random(seed)

    def reset(self):
        self.acc = 0
        self.fires = 0

    def step(self, fw):
        noise = int(self._rng.gauss(0, TH // 4))
        self.acc += fw + noise
        if self.acc > self.th:
            self.fires += 1
            self.acc -= self.offset
        elif self.acc < -self.th:
            self.acc += self.offset

# ========== HD NCO Array ==========
class HDNCOArray:
    def __init__(self, N=N, n_steps=N_STEPS):
        self.N = N
        self.n_steps = n_steps
        rng = random.Random(1)
        # Weights: wide range for diverse selectivity
        self.weights = [rng.uniform(0.02, 3.0) for _ in range(N)]
        # Diverse thresholds
        self.ncos = []
        for i in range(N):
            th_var = int(TH * (0.3 + 1.7 * i / N))
            self.ncos.append(NCO(th_var, OFFSET, seed=1000 + i * 37))
        # Readout: C x N (learned)
        self.W = None

    def fingerprint(self, signal_784):
        """
        signal_784: list of 784 floats in [0, 1]
        Returns: list of N firing counts (each int 0..N_STEPS)
        """
        f = []
        for i in range(N):
            nco = self.ncos[i]
            nco.reset()
            # Sum pixel values weighted by W_i, then normalize
            # Use mean pixel intensity * weight as frequency word
            avg_signal = signal_784  # already a float (pre-computed mean)
            fw = int(avg_signal * self.weights[i] * TH // 20)
            # Clamp: avoid overflow
            if fw > TH:
                fw = TH
            for _ in range(self.n_steps):
                nco.step(fw)
            f.append(nco.fires)
        return f

    def fingerprint_pixelweighted(self, signal_784):
        """
        Version 2: each NCO has its own pixel template (random projection).
        F_word = sum over 784 pixels of pixel_i * template_i * weight_i / scale
        This is the true HD approach: each NCO computes a random projection of the
        input image, then integrates over time.
        """
        f = []
        # Pre-compute template weights for each NCO (random projection)
        for i in range(N):
            nco = self.ncos[i]
            nco.reset()
            seed_i = 2000 + i * 13
            rng_i = random.Random(seed_i)
            template = [rng_i.uniform(-1.0, 1.0) for _ in range(784)]
            dot = sum(s * t for s, t in zip(signal_784, template))
            # Scale to frequency word
            scale = 784 * 0.5  # typical dot product magnitude
            fw = int(dot * self.weights[i] * TH // int(scale * N_STEPS))
            if fw > TH:
                fw = TH
            elif fw < 0:
                fw = 0
            for _ in range(self.n_steps):
                nco.step(fw)
            f.append(nco.fires)
        return f

    def train_ridge(self, X, y):
        """
        Ridge Regression: W_out = (X^T X + alpha I)^{-1} X^T T
        X: M x N fingerprint matrix
        y: M labels (0..C-1)
        T: M x C one-hot

        Solved via normal equations (no numpy).
        Uses Cholesky-like iterative solver for stability.
        """
        M = len(X)
        # One-hot targets
        T = [[1.0 if j == y[i] else 0.0 for j in range(C)] for i in range(M)]

        # Build Gram matrix G = X^T X + alpha I  (N x N)
        G = [[0.0] * N for _ in range(N)]
        for i in range(N):
            G[i][i] = ALPHA
            for k in range(M):
                G[i][i] += X[k][i] * X[k][i]
            for j in range(i + 1, N):
                val = sum(X[k][i] * X[k][j] for k in range(M))
                G[i][j] = val
                G[j][i] = val

        # Build RHS B = X^T T  (N x C)
        B = [[0.0] * C for _ in range(N)]
        for i in range(N):
            for j in range(C):
                B[i][j] = sum(X[k][i] * T[k][j] for k in range(M))

        # Solve G @ W = B for each class using conjugate gradient
        W = [[0.0] * N for _ in range(C)]
        for j in range(C):
            b = [B[i][j] for i in range(N)]
            w = self._cg_solve(G, b, max_iter=200, tol=1e-6)
            for i in range(N):
                W[j][i] = w[i]

        self.W = W  # C x N

    def _cg_solve(self, A, b, max_iter=200, tol=1e-6):
        """Conjugate gradient for N x N system."""
        n = len(b)
        x = [0.0] * n
        r = b[:]
        p = b[:]
        rr = sum(ri * ri for ri in r)
        if rr < tol * tol:
            return x
        for _ in range(max_iter):
            # Ap = A @ p
            Ap = [0.0] * n
            for i in range(n):
                s = 0.0
                for j in range(n):
                    s += A[i][j] * p[j]
                Ap[i] = s
            pAp = sum(p[i] * Ap[i] for i in range(n))
            if pAp == 0.0:
                break
            alpha = rr / pAp
            for i in range(n):
                x[i] += alpha * p[i]
                r[i] -= alpha * Ap[i]
            rr_new = sum(ri * ri for ri in r)
            if rr_new < tol * tol:
                break
            beta = rr_new / rr
            for i in range(n):
                p[i] = r[i] + beta * p[i]
            rr = rr_new
        return x

    def predict(self, f):
        """Classify from fingerprint vector."""
        scores = [sum(self.W[j][k] * f[k] for k in range(N)) for j in range(C)]
        max_s = max(scores)
        candidates = [j for j, s in enumerate(scores) if s == max_s]
        return candidates[0] if len(candidates) == 1 else candidates[0]


# ========== MNIST Loader (pure Python — no TF/Keras) ==========
import gzip
import struct

def load_mnist(images_path, labels_path, max_count=None):
    """Load MNIST from gzipped IDX files. Returns (images, labels)."""
    with gzip.open(labels_path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        labels = list(f.read(n))
    with gzip.open(images_path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        images = []
        for i in range(n):
            img_data = list(f.read(rows * cols))
            # Normalize to [0, 1]
            img_norm = [p / 255.0 for p in img_data]
            images.append(img_norm)

    if max_count is not None and max_count < len(images):
        images = images[:max_count]
        labels = labels[:max_count]

    return images, labels


# ========== Main ==========
def main():
    sep = "=" * 60
    print(sep)
    print("  HD NCO Array — MNIST Prototype")
    print(sep)
    print(f"  NCOs:               {N}")
    print(f"  Classes:            {C} (digits 0-9)")
    print(f"  N_STEPS:            {N_STEPS}")
    print(f"  Input dimension:    784 (28x28)")
    print(f"  Ridge alpha:        {ALPHA}")
    print(f"  Train samples:      {TRAIN_SIZE}")
    print(f"  Test samples:       {TEST_SIZE}")
    print()

    # Load MNIST
    print("  Loading MNIST...", end=" ", flush=True)
    base = "."
    X_train, y_train = load_mnist(
        f"{base}/train-images-idx3-ubyte.gz",
        f"{base}/train-labels-idx1-ubyte.gz",
        max_count=TRAIN_SIZE
    )
    X_test, y_test = load_mnist(
        f"{base}/t10k-images-idx3-ubyte.gz",
        f"{base}/t10k-labels-idx1-ubyte.gz",
        max_count=TEST_SIZE
    )
    print(f"done. ({len(X_train)} train, {len(X_test)} test)")

    # Create array
    hd = HDNCOArray(N, N_STEPS)
    print()

    # === Fingerprint Phase ===
    print("  Computing train fingerprints...", end=" ", flush=True)
    t0 = time.time()
    # Quick mode: use mean intensity (faster)
    X_train_fp = [hd.fingerprint(img) for img in X_train]
    t_train_fp = time.time() - t0
    print(f"done. ({t_train_fp:.1f}s, {len(X_train_fp)/t_train_fp:.0f} img/s)")

    # === Training Phase ===
    print("  Training Ridge readout...", end=" ", flush=True)
    t0 = time.time()
    # Normalize fingerprints to [0,1] for better numerical stability
    max_fire = N_STEPS
    X_train_norm = [[v / max_fire for v in fp] for fp in X_train_fp]
    hd.train_ridge(X_train_norm, y_train)
    t_train = time.time() - t0
    print(f"done. ({t_train:.1f}s)")

    # === Test Phase ===
    print("  Computing test fingerprints...", end=" ", flush=True)
    t0 = time.time()
    X_test_fp = [hd.fingerprint(img) for img in X_test]
    t_test_fp = time.time() - t0
    print(f"done. ({t_test_fp:.1f}s, {len(X_test_fp)/t_test_fp:.0f} img/s)")

    print("  Classifying...", end=" ", flush=True)
    t0 = time.time()
    correct = 0
    total = len(X_test)
    cm = [[0] * C for _ in range(C)]
    for i in range(total):
        f_norm = [v / max_fire for v in X_test_fp[i]]
        pred = hd.predict(f_norm)
        cm[y_test[i]][pred] += 1
        if pred == y_test[i]:
            correct += 1
    t_classify = time.time() - t0
    accuracy = correct / total * 100

    print(f"done. ({t_classify:.1f}s)")
    print()
    print(f"  🔵 Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print()

    # Per-class accuracy
    print("  Per-class Accuracy:")
    for c in range(C):
        total_c = sum(cm[c])
        correct_c = cm[c][c]
        print(f"    Digit {c}: {correct_c}/{total_c} = {correct_c/total_c*100:.1f}%")

    print()
    print("  Confusion Matrix (rows=true, cols=pred):")
    header = " " * 8 + "".join(f"{j:>6}" for j in range(C))
    print(header)
    for c in range(C):
        row = f"  {c:>4}  "
        for p in range(C):
            row += f"{cm[c][p]:>6}"
        print(row)

    print()
    print("  Performance Summary:")
    print(f"    Train FP time:  {t_train_fp:.1f}s ({len(X_train)/t_train_fp:.0f} img/s)")
    print(f"    Train Ridge:    {t_train:.1f}s")
    print(f"    Test FP time:   {t_test_fp:.1f}s ({len(X_test)/t_test_fp:.0f} img/s)")
    print(f"    Classify time:  {t_classify:.1f}s")
    total_time = t_train_fp + t_train + t_test_fp + t_classify
    print(f"    Total time:     {total_time:.1f}s")

    # Fingerprint stats
    print()
    print("  Fingerprint Statistics (per digit):")
    for digit in range(C):
        fps_digit = [X_test_fp[i] for i in range(total) if y_test[i] == digit][:100]
        if not fps_digit:
            continue
        mean_f = [sum(fp[k] for fp in fps_digit) / len(fps_digit) for k in range(N)]
        nonzero = sum(1 for v in mean_f if v > 0.01)
        avg_fire = sum(mean_f) / N
        print(f"    Digit {digit}: {nonzero}/{N} NCOs fire, avg fire = {avg_fire:.3f}")

    print()
    print(sep)
    print("✅ MNIST prototype complete.")
    print(sep)


if __name__ == "__main__":
    main()
