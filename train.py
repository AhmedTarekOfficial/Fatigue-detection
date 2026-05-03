# ============================================================
#  train.py  —  Training Pipeline for all 4 models
#
#  Usage:
#      python train.py --model eye
#      python train.py --model mouth
#      python train.py --model head
#      python train.py --model microsleep
#      python train.py --model all
# ============================================================

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

from config import *
from data_loader import (load_dataset, split_dataset, build_generators,
                         compute_class_weights, load_sequences)
from models import (build_eye_model, build_mouth_model,
                    build_head_model, build_microsleep_model,
                    unfreeze_top_layers)


# ──────────────────────────────────────────────────────────────
# CALLBACKS
# ──────────────────────────────────────────────────────────────
def build_callbacks(model_name: str):
    """
    Callbacks:
        ModelCheckpoint → يحفظ أحسن model بناءً على val_loss
        EarlyStopping   → يوقف الـ training لو الـ val_loss بطلت تتحسن
        ReduceLROnPlateau → يقلل الـ LR لو الـ loss وقفت
        TensorBoard     → لـ visualization (optional)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join(MODELS_DIR, f"{model_name}_{timestamp}.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,            # استنى 7 epochs قبل ما توقف
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,            # بتنص الـ LR
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOGS_DIR, f"{model_name}_{timestamp}"),
            histogram_freq=1
        ),
    ]

    return callbacks, save_path


# ──────────────────────────────────────────────────────────────
# TRAINING FUNCTION
# ──────────────────────────────────────────────────────────────
def train_model(model, train_gen, val_gen, model_name: str,
                class_weights=None, epochs=EPOCHS):
    """
    Phase 1: Train مع frozen backbone
    Phase 2: Fine-tune مع unfrozen top layers
    """
    callbacks, save_path = build_callbacks(model_name)

    print(f"\n{'='*55}")
    print(f"  PHASE 1 — Training {model_name} (frozen backbone)")
    print(f"{'='*55}")

    history_1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # ── Phase 2: Fine-tuning ───────────────────────────────
    print(f"\n{'='*55}")
    print(f"  PHASE 2 — Fine-tuning {model_name} (unfrozen top layers)")
    print(f"{'='*55}")

    model = unfreeze_top_layers(model, n_layers_to_unfreeze=30)

    history_2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,             # fine-tune أقل epochs
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    print(f"\n  ✓ Best model saved → {save_path}")
    return model, history_1, history_2, save_path


# ──────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, class_names: list, model_name: str):
    """
    تقييم شامل على الـ test set:
        - Accuracy, AUC, Precision, Recall, F1
        - Confusion Matrix
        - Classification Report
    """
    from sklearn.metrics import (classification_report,
                                  confusion_matrix,
                                  ConfusionMatrixDisplay)

    print(f"\n{'='*55}")
    print(f"  EVALUATION — {model_name}")
    print(f"{'='*55}")

    y_test_cat = tf.keras.utils.to_categorical(y_test, len(class_names))
    results = model.evaluate(X_test, y_test_cat, verbose=0)

    for name, val in zip(model.metrics_names, results):
        print(f"  {name:15s}: {val:.4f}")

    # Predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(LOGS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"\n  ✓ Confusion matrix saved → {cm_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# PLOT TRAINING HISTORY
# ──────────────────────────────────────────────────────────────
def plot_history(history_1, history_2, model_name: str):
    # Merge the two phases
    metrics = {}
    for key in history_1.history:
        metrics[key] = history_1.history[key] + history_2.history.get(key, [])

    epochs = range(1, len(metrics["loss"]) + 1)
    fine_start = len(history_1.history["loss"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{model_name} — Training History", fontsize=13)

    # Loss
    axes[0].plot(epochs, metrics["loss"],     label="Train Loss")
    axes[0].plot(epochs, metrics["val_loss"], label="Val Loss")
    axes[0].axvline(fine_start, color="gray", linestyle="--", label="Fine-tune start")
    axes[0].set_title("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, metrics["accuracy"],     label="Train Acc")
    axes[1].plot(epochs, metrics["val_accuracy"], label="Val Acc")
    axes[1].axvline(fine_start, color="gray", linestyle="--", label="Fine-tune start")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(LOGS_DIR, f"{model_name}_history.png")
    plt.savefig(plot_path, dpi=150)
    print(f"  ✓ Training plot saved → {plot_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# MAIN RUNNERS — لكل model
# ──────────────────────────────────────────────────────────────
def run_eye():
    print("\n>>> Training: EyeFatigueCNN")
    X, y = load_dataset(EYE_DATA_DIR, EYE_CLASSES)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y)

    class_weights = compute_class_weights(y_train)
    train_gen, val_gen = build_generators(X_train, y_train, X_val, y_val, n_classes=2)

    model = build_eye_model()
    model, h1, h2, _ = train_model(model, train_gen, val_gen, "eye_model",
                                   class_weights=class_weights)
    evaluate_model(model, X_test, y_test, EYE_CLASSES, "EyeFatigueCNN")
    plot_history(h1, h2, "EyeFatigueCNN")


def run_mouth():
    print("\n>>> Training: YawningCNN")
    X, y = load_dataset(MOUTH_DATA_DIR, MOUTH_CLASSES)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y)

    class_weights = compute_class_weights(y_train)
    train_gen, val_gen = build_generators(X_train, y_train, X_val, y_val, n_classes=2)

    model = build_mouth_model()
    model, h1, h2, _ = train_model(model, train_gen, val_gen, "mouth_model",
                                   class_weights=class_weights)
    evaluate_model(model, X_test, y_test, MOUTH_CLASSES, "YawningCNN")
    plot_history(h1, h2, "YawningCNN")


def run_head():
    print("\n>>> Training: HeadNoddingCNN")
    X, y = load_dataset(HEAD_DATA_DIR, HEAD_CLASSES)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y)

    class_weights = compute_class_weights(y_train)
    train_gen, val_gen = build_generators(X_train, y_train, X_val, y_val, n_classes=2)

    model = build_head_model()
    model, h1, h2, _ = train_model(model, train_gen, val_gen, "head_model",
                                   class_weights=class_weights)
    evaluate_model(model, X_test, y_test, HEAD_CLASSES, "HeadNoddingCNN")
    plot_history(h1, h2, "HeadNoddingCNN")


def run_microsleep():
    print("\n>>> Training: MicrosleepModel (CNN + LSTM)")
    X_seq, y_seq = load_sequences(SEQ_DATA_DIR, seq_len=MICROSLEEP_FRAMES)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X_seq, y_seq)

    y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
    y_val_cat   = tf.keras.utils.to_categorical(y_val,   2)

    model = build_microsleep_model(seq_len=MICROSLEEP_FRAMES)
    callbacks, _ = build_callbacks("microsleep_model")

    h1 = model.fit(X_train, y_train_cat,
                   validation_data=(X_val, y_val_cat),
                   epochs=EPOCHS,
                   batch_size=8,    # أصغر batch لأن الـ sequences ثقيلة
                   callbacks=callbacks,
                   verbose=1)

    evaluate_model(model, X_test, y_test, ["normal", "microsleep"], "MicrosleepModel")


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fatigue Detection Models")
    parser.add_argument(
        "--model",
        choices=["eye", "mouth", "head", "microsleep", "all"],
        default="all",
        help="Which model to train"
    )
    args = parser.parse_args()

    runners = {
        "eye":        run_eye,
        "mouth":      run_mouth,
        "head":       run_head,
        "microsleep": run_microsleep,
    }

    if args.model == "all":
        for name, fn in runners.items():
            fn()
    else:
        runners[args.model]()

    print("\n✓ Training complete!")
