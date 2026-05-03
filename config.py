# ============================================================
#  config.py  —  Central configuration for all models
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")          # ← ضع داتاك هنا
MODELS_DIR      = os.path.join(BASE_DIR, "saved_models")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

# Dataset sub-folders  (كل فولدر فيه: open/ و closed/ أو yawning/ و not_yawning/ إلخ)
EYE_DATA_DIR    = os.path.join(DATA_DIR, "eye")           # open / closed
MOUTH_DATA_DIR  = os.path.join(DATA_DIR, "mouth")         # yawning / not_yawning
HEAD_DATA_DIR   = os.path.join(DATA_DIR, "head")          # nodding / alert
SEQ_DATA_DIR    = os.path.join(DATA_DIR, "sequences")     # for LSTM (video clips)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

# ── Image settings ───────────────────────────────────────────
IMG_SIZE        = (224, 224)   # MobileNetV2 / EfficientNet input
IMG_CHANNELS    = 3
INPUT_SHAPE     = (*IMG_SIZE, IMG_CHANNELS)

# ── Training hyperparameters ─────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 30
LEARNING_RATE   = 1e-4
FINE_TUNE_LR    = 1e-5         # بعد الـ freeze
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15
SEED            = 42

# ── Augmentation ─────────────────────────────────────────────
AUGMENT = dict(
    rotation_range      = 15,
    width_shift_range   = 0.1,
    height_shift_range  = 0.1,
    zoom_range          = 0.15,
    horizontal_flip     = True,
    brightness_range    = [0.7, 1.3],
    fill_mode           = "nearest",
)

# ── MediaPipe ─────────────────────────────────────────────────
FACE_CONFIDENCE     = 0.5
LANDMARK_CONFIDENCE = 0.5

# Eye landmark indices (MediaPipe 468-point mesh)
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# Mouth landmark indices
MOUTH_IDX = [61, 291, 81, 178, 13, 14, 311, 402]

# ── Thresholds (Inference) ────────────────────────────────────
EAR_THRESHOLD        = 0.25    # Eye Aspect Ratio  — أقل = عين مغلقة
MAR_THRESHOLD        = 0.6     # Mouth Aspect Ratio — أكبر = تثاؤب
PITCH_THRESHOLD      = 20      # درجة ميل الرأس
PERCLOS_THRESHOLD    = 0.7     # % الوقت اللي العين فيه مغلقة
TEMPORAL_WINDOW      = 60      # frames للـ smoothing
MICROSLEEP_FRAMES    = 30      # 1 ثانية @ 30fps = microsleep

# ── Class names ──────────────────────────────────────────────
EYE_CLASSES   = ["open", "closed"]
MOUTH_CLASSES = ["not_yawning", "yawning"]
HEAD_CLASSES  = ["alert", "nodding"]
