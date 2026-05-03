# ============================================================
#  data_loader.py  —  Dataset loading + Augmentation pipeline
#  Assumes user's data is already organized in folders:
#      data/eye/open/   data/eye/closed/
#      data/mouth/not_yawning/   data/mouth/yawning/
#      data/head/alert/   data/head/nodding/
# ============================================================

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import *


# ──────────────────────────────────────────────────────────────
# 1.  LOAD IMAGES FROM FOLDER STRUCTURE
# ──────────────────────────────────────────────────────────────
def load_dataset(data_dir: str, class_names: list, img_size=IMG_SIZE):
    """
    يقرأ الصور من فولدرات مرتبة زي:
        data_dir/
            class_0/   ← الصور
            class_1/   ← الصور

    Returns: X (numpy array), y (numpy array), class_names (list)
    """
    X, y = [], []

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Folder not found: {class_dir}")

        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        print(f"  [{class_name}]  {len(images)} images")

        for img_file in images:
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # BGR → RGB  +  Resize
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(label)

    X = np.array(X, dtype=np.float32) / 255.0   # Normalize [0, 1]
    y = np.array(y, dtype=np.int32)
    return X, y


# ──────────────────────────────────────────────────────────────
# 2.  SPLIT  →  Train / Val / Test
# ──────────────────────────────────────────────────────────────
def split_dataset(X, y, val_split=VAL_SPLIT, test_split=TEST_SPLIT, seed=SEED):
    """
    Split الداتا:
        Train : 70%
        Val   : 15%
        Test  : 15%
    """
    # أولاً نفصل الـ test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_split,
        random_state=seed,
        stratify=y          # نضمن توزيع متوازن لكل class
    )

    # تاني نفصل الـ val من الـ train
    val_ratio = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_train_val
    )

    print(f"\n  Train:  {len(X_train)} samples")
    print(f"  Val:    {len(X_val)} samples")
    print(f"  Test:   {len(X_test)} samples\n")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ──────────────────────────────────────────────────────────────
# 3.  AUGMENTATION  →  ImageDataGenerator
# ──────────────────────────────────────────────────────────────
def build_generators(X_train, y_train, X_val, y_val, n_classes: int):
    """
    Train generator: مع augmentation عشان نزود التنوع ونمنع الـ overfitting
    Val generator:   بدون augmentation (بنعمل evaluate بالصور الأصلية)
    """
    # Train — مع augmentation
    train_datagen = ImageDataGenerator(**AUGMENT)

    # Val — بس rescale (خالص عملنا normalize يدوياً)
    val_datagen = ImageDataGenerator()

    # One-hot encode اللو عندنا أكتر من class
    y_train_cat = tf.keras.utils.to_categorical(y_train, n_classes)
    y_val_cat   = tf.keras.utils.to_categorical(y_val,   n_classes)

    train_gen = train_datagen.flow(
        X_train, y_train_cat,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )
    val_gen = val_datagen.flow(
        X_val, y_val_cat,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_gen, val_gen


# ──────────────────────────────────────────────────────────────
# 4.  CLASS WEIGHTS  (لو الداتا مش متوازنة)
# ──────────────────────────────────────────────────────────────
def compute_class_weights(y):
    """
    لو عندك مثلاً 80% open eyes و 20% closed —
    الـ model هيـ bias للـ majority.
    الـ weights دي بتعوّض الـ imbalance.
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


# ──────────────────────────────────────────────────────────────
# 5.  SEQUENCE LOADER  (للـ Microsleep LSTM)
# ──────────────────────────────────────────────────────────────
def load_sequences(seq_dir: str, seq_len=MICROSLEEP_FRAMES, img_size=IMG_SIZE):
    """
    يقرأ فيديو clips كـ sequences من frames:
        seq_dir/
            microsleep/    ← فيديوهات .mp4 أو مجلدات frames
            normal/

    Returns: X_seq (N, seq_len, H, W, 3), y_seq (N,)
    """
    X_seq, y_seq = [], []
    class_names = ["normal", "microsleep"]

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(seq_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  ⚠ Sequence folder not found: {class_dir} — skipping")
            continue

        clips = [f for f in os.listdir(class_dir)
                 if f.lower().endswith(('.mp4', '.avi', '.mov'))]

        for clip_file in clips:
            clip_path = os.path.join(class_dir, clip_file)
            frames = _extract_frames(clip_path, seq_len, img_size)
            if frames is not None:
                X_seq.append(frames)
                y_seq.append(label)

    X_seq = np.array(X_seq, dtype=np.float32) / 255.0
    y_seq = np.array(y_seq, dtype=np.int32)
    return X_seq, y_seq


def _extract_frames(video_path: str, n_frames: int, img_size: tuple):
    """يستخرج n_frames متساوية المسافة من فيديو"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n_frames:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, img_size)
        frames.append(frame)

    cap.release()
    return np.array(frames)
