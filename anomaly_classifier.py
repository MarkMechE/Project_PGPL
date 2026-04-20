# anomaly_classifier.py
# Trains CNN-MLP on synthetic Busan anomaly signals
# Run: python anomaly_classifier.py
# Output: outputs/anomaly_classifier.pkl

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import pickle

os.makedirs("outputs", exist_ok=True)

FS   = 512   # FFT bins
SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------------ #
#  Generate synthetic Busan water distribution anomalies              #
# ------------------------------------------------------------------ #
def make_sample(label):
    t = np.linspace(0, 1, FS)
    noise = 0.05 * np.random.randn(FS)
    if label == "leak":
        # 1-5 kHz steady hiss — Gaussian peak at mid-freq
        sig = 1.5 * np.exp(-((t * 5 - 2.5) ** 2)) + noise
    elif label == "burst":
        # Broadband impulse — exponential decay
        sig = np.random.exponential(3, FS) * np.exp(-t * 10) + noise
    elif label == "drop":
        # Low-freq pressure throb — damped sine
        sig = 0.5 * np.sin(2 * np.pi * 80 * t) * np.exp(-t * 2) + noise
    else:  # normal
        # Tidal rumble — low-amp low-freq
        sig = 0.3 * np.sin(2 * np.pi * 40 * t) + noise * 0.5
    return sig

def generate_dataset(n_per_class=2500):
    labels_list = ["leak", "burst", "drop", "normal"]
    X, y = [], []
    for lbl in labels_list:
        for _ in range(n_per_class):
            X.append(make_sample(lbl))
            y.append(lbl)
    return np.array(X), np.array(y)

print("Generating synthetic Busan anomaly dataset...")
X, y_str = generate_dataset(n_per_class=2500)
le = LabelEncoder()
y  = le.fit_transform(y_str)
print(f"  Dataset: {X.shape[0]} samples x {X.shape[1]} features")
print(f"  Classes: {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# ------------------------------------------------------------------ #
#  MLP classifier (sklearn — no TF dep issues, fast, exportable)     #
# ------------------------------------------------------------------ #
print("\nTraining MLP classifier...")
clf = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    max_iter=50,
    random_state=SEED,
    verbose=True,
    early_stopping=True,
    validation_fraction=0.1,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

acc = (y_pred == y_test).mean()
print(f"Test accuracy: {acc:.3f}")

# ------------------------------------------------------------------ #
#  Save model + encoder                                               #
# ------------------------------------------------------------------ #
bundle = {"model": clf, "encoder": le, "feature_size": FS}
with open("outputs/anomaly_classifier.pkl", "wb") as f:
    pickle.dump(bundle, f)
print("\nSaved: outputs/anomaly_classifier.pkl")