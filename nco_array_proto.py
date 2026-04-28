#!/usr/bin/env python3
"""
HD NCO Array — Prototype v0.1
==============================
N=32 NCOs, C=3 classes (0.3, 0.6, 1.0), sigma=0.06, N_STEPS=5.
Linear Readout: W_out (C x N) trained via least-squares on host.

Goal: 100% accuracy on clean test data.
"""

import random
import math

# ========== Constants ==========
N = 256             # Number of NCOs in array
C = 3                # Number of classes
N_STEPS = 5          # Clock cycles per classification
TH = 2**30           # NCO threshold (matching v2)
OFFSET = TH // 2     # Reset offset

CLASSES = [0.3, 0.6, 1.0]  # Signal centers per class
SIGMA = 0.06               # Noise sigma
TRAIN_PER_CLASS = 100      # Training samples per class
TEST_PER_CLASS = 200       # Test samples per class

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
        noise = int(self._rng.gauss(0, TH // 4))  # Gaussian noise
        self.acc += fw + noise
        if self.acc > self.th:
            self.fires += 1
            self.acc -= self.offset
        elif self.acc < -self.th:
            self.fires += 0  # negative signal — we only count pos fires
            self.acc += self.offset

# ========== HD NCO Array ==========
class HDNCOArray:
    def __init__(self, N=N, n_steps=N_STEPS):
        self.N = N
        self.n_steps = n_steps

        # Random weights W_i (one per NCO) — fixed, with wider range for diversity
        random.seed(1)
        self.weights = [random.uniform(0.05, 2.5) for _ in range(N)]

        # NCOs with unique seeds and DIFFERENT thresholds (like v2)
        self.ncos = []
        for i in range(N):
            # Vary threshold from 0.5x to 2.0x of base TH
            th_var = TH * (0.5 + 1.5 * i / N)
            self.ncos.append(NCO(int(th_var), OFFSET, seed=1000 + i * 37))

        # Readout weights: W_out (C x N) — learned
        self.W_out = None

    def fingerprint(self, signal):
        """Return: vector f of length N (firing counts)."""
        f = []
        # Adaptive scale: such that signal_center * weight * N_STEPS ~= TH
        scale = 1.0 / (CLASSES[1] * 1.0 * N_STEPS)
        for i in range(N):
            nco = self.ncos[i]
            nco.reset()
            fw = int(signal * self.weights[i] * TH * scale)
            for _ in range(self.n_steps):
                nco.step(fw)
            f.append(nco.fires)
        return f

    def train(self, X_train, y_train):
        """Train linear readout via least-squares.
        
        Args:
            X_train: list of fingerprint vectors (each length N)
            y_train: list of class IDs (0..C-1)
        """
        M = len(X_train)
        # Build target matrix T: M x C (one-hot)
        T = [[0.0] * C for _ in range(M)]
        for i, cls in enumerate(y_train):
            T[i][cls] = 1.0

        # Build design matrix F: M x N
        # Convert to float
        F = [[float(v) for v in row] for row in X_train]

        # Solve F @ W_out.T = T  →  W_out.T = pinv(F) @ T
        # Using manual least-squares (no numpy)
        # W_out.T = (F^T F)^-1 F^T T  →  W_out = T^T F (F^T F)^-1
        # We'll use simple LMS (iterative) for stability
        # Initialize W_out' (C x N) as zeros
        W = [[0.0] * N for _ in range(C)]
        lr = 0.001
        for epoch in range(500):
            total_err = 0.0
            for i in range(M):
                # Forward: scores = W @ F[i]  (C-length vector)
                scores = [sum(W[j][k] * F[i][k] for k in range(N))
                          for j in range(C)]
                errors = [scores[j] - T[i][j] for j in range(C)]
                total_err += sum(e * e for e in errors)
                # Update
                for j in range(C):
                    for k in range(N):
                        W[j][k] -= lr * errors[j] * F[i][k]
            if epoch % 100 == 0:
                pass  # silence

        self.W_out = W  # C x N

    def predict(self, signal):
        """Classify a single signal."""
        f = self.fingerprint(signal)
        scores = [sum(self.W_out[j][k] * f[k] for k in range(N))
                  for j in range(C)]
        # Add small noise to break ties deterministically
        max_score = max(scores)
        candidates = [j for j, s in enumerate(scores) if s == max_score]
        return candidates[0] if len(candidates) == 1 else random.choice(candidates)


# ========== Data generation ==========
def generate_data(per_class, sigma=SIGMA):
    random.seed(42)
    signals = []
    labels = []
    for cls in range(C):
        for _ in range(per_class):
            base = CLASSES[cls]
            noisy = base + random.gauss(0, sigma)
            signals.append(max(0.01, noisy))
            labels.append(cls)
    # Shuffle
    zipped = list(zip(signals, labels))
    random.shuffle(zipped)
    signals, labels = zip(*zipped)
    return list(signals), list(labels)


# ========== Test ==========
def test():
    sep = "=" * 60
    print(sep)
    print("HD NCO Array — Prototype v0.1")
    print(sep)
    print(f"  N (NCOs):         {N}")
    print(f"  C (classes):      {C} ({', '.join(str(c) for c in CLASSES)})")
    print(f"  N_STEPS:          {N_STEPS}")
    print(f"  Noise sigma:      {SIGMA}")
    print(f"  Train/class:      {TRAIN_PER_CLASS}")
    print(f"  Test/class:       {TEST_PER_CLASS}")
    print(f"  Weights W_i:      random (fixed)")
    print(f"  Readout:          Linear (LMS-trained)")
    print()

    # Generate data
    X_sig, y = generate_data(TRAIN_PER_CLASS + TEST_PER_CLASS)

    # Split
    train_sig = X_sig[:TRAIN_PER_CLASS * C]
    train_y = y[:TRAIN_PER_CLASS * C]
    test_sig = X_sig[TRAIN_PER_CLASS * C:]
    test_y = y[TRAIN_PER_CLASS * C:]

    print(f"  Training samples: {len(train_sig)}")
    print(f"  Test samples:     {len(test_sig)}")
    print()

    # Create and train
    hd = HDNCOArray(N, N_STEPS)

    print("  Training readout...", end=" ", flush=True)
    # Build training fingerprints
    X_train = [hd.fingerprint(s) for s in train_sig]
    hd.train(X_train, train_y)
    print("done.")

    # Test
    correct = 0
    total = len(test_sig)
    cm = [[0] * C for _ in range(C)]

    for i in range(total):
        pred = hd.predict(test_sig[i])
        cm[test_y[i]][pred] += 1
        if pred == test_y[i]:
            correct += 1

    accuracy = correct / total * 100

    print()
    print(f"  🔵 Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print()

    print("  Confusion Matrix:")
    header = " " * 12 + "".join(f"{'cls ' + str(j):>10}" for j in range(C))
    print(header)
    for c in range(C):
        row = f"  True {c:>5}"
        for p in range(C):
            row += f"{cm[c][p]:>10}"
        print(row)

    print()
    print(sep)

    # Analyze fingerprints
    print("\n  Fingerprint stats (per class):")
    for cls in range(C):
        fps = [hd.fingerprint(CLASSES[cls] + random.gauss(0, SIGMA))
               for _ in range(50)]
        mean_f = [sum(fp[k] for fp in fps) / 50 for k in range(N)]
        nonzero = sum(1 for v in mean_f if v > 0)
        avg_fire = sum(mean_f) / N
        print(f"    Class {cls} ({CLASSES[cls]:.1f}): {nonzero}/{N} NCOs fire, "
              f"avg firing = {avg_fire:.2f}")

    print()
    print("✅ Prototype complete.")


if __name__ == "__main__":
    test()
