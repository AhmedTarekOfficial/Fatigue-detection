# ============================================================
#  models.py  —  CNN Architectures for Fatigue Detection
#
#  ① EyeFatigueCNN   → MobileNetV2 + custom head
#  ② YawningCNN      → MobileNetV2 + custom head
#  ③ HeadNoddingCNN  → EfficientNetB0 + custom head
#  ④ MicrosleepModel → CNN features + LSTM
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.regularizers import l2
from config import INPUT_SHAPE, LEARNING_RATE, FINE_TUNE_LR


# ──────────────────────────────────────────────────────────────
# HELPER: Custom Classification Head
# ──────────────────────────────────────────────────────────────
def _classification_head(x, n_classes: int, dropout_rate=0.4, name_prefix=""):
    """
    Head مشترك بين الـ models:
        GlobalAveragePooling → Dense(256) → BN → Dropout → Output

    GlobalAveragePooling بدل Flatten:
        - بتاخد متوسط كل feature map → vector صغير
        - أقل parameters → أقل overfitting
        - أسرع في الـ inference
    """
    x = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)

    x = layers.Dense(256, activation="relu",
                      kernel_regularizer=l2(1e-4),
                      name=f"{name_prefix}_dense1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop1")(x)

    x = layers.Dense(128, activation="relu",
                      kernel_regularizer=l2(1e-4),
                      name=f"{name_prefix}_dense2")(x)
    x = layers.Dropout(dropout_rate / 2, name=f"{name_prefix}_drop2")(x)

    activation = "sigmoid" if n_classes == 2 else "softmax"
    output_units = 1 if n_classes == 2 else n_classes
    out = layers.Dense(output_units, activation=activation,
                       name=f"{name_prefix}_output")(x)
    return out


# ──────────────────────────────────────────────────────────────
# ①  EyeFatigueCNN  —  MobileNetV2
# ──────────────────────────────────────────────────────────────
def build_eye_model(input_shape=INPUT_SHAPE, n_classes=2):
    """
    يكتشف إغماض العين (open / closed)
    Input: Eye ROI crop (224×224×3)

    Architecture:
        MobileNetV2 (pretrained ImageNet, frozen at first)
        → Custom head
        → sigmoid output

    MobileNetV2 مناسب لأنه:
        - خفيف جداً (3.4M params)
        - سريع في الـ inference
        - Depthwise Separable Convolutions
    """
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,        # بنحذف الـ classifier الأصلي
        weights="imagenet",       # Pretrained features من ImageNet
        alpha=1.0                 # Width multiplier
    )
    base.trainable = False        # Phase 1: freeze الـ base

    inputs = Input(shape=input_shape, name="eye_input")

    # Data Augmentation داخل الـ model (بيشتغل بس في الـ training)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomBrightness(0.2)(x)
    x = layers.RandomContrast(0.2)(x)

    # Preprocess للـ MobileNetV2 (scale من [0,1] إلى [-1,1])
    x = layers.Lambda(lambda t: t * 2.0 - 1.0, name="preprocess")(x)

    x = base(x, training=False)   # training=False → BN يبقى في eval mode
    output = _classification_head(x, n_classes, name_prefix="eye")

    model = Model(inputs, output, name="EyeFatigueCNN")
    _compile_model(model, n_classes, lr=LEARNING_RATE)
    return model


# ──────────────────────────────────────────────────────────────
# ②  YawningCNN  —  MobileNetV2
# ──────────────────────────────────────────────────────────────
def build_mouth_model(input_shape=INPUT_SHAPE, n_classes=2):
    """
    يكتشف التثاؤب (not_yawning / yawning)
    Input: Mouth ROI crop (224×224×3)

    نفس الـ architecture بتاع Eye بس:
        - Input: منطقة الفم بدل العين
        - أضفنا RandomRotation لأن الفم بيتحرك أكتر
    """
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        alpha=1.0
    )
    base.trainable = False

    inputs = Input(shape=input_shape, name="mouth_input")
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomBrightness(0.2)(x)
    x = layers.Lambda(lambda t: t * 2.0 - 1.0, name="preprocess")(x)

    x = base(x, training=False)
    output = _classification_head(x, n_classes, name_prefix="mouth")

    model = Model(inputs, output, name="YawningCNN")
    _compile_model(model, n_classes, lr=LEARNING_RATE)
    return model


# ──────────────────────────────────────────────────────────────
# ③  HeadNoddingCNN  —  EfficientNetB0
# ──────────────────────────────────────────────────────────────
def build_head_model(input_shape=INPUT_SHAPE, n_classes=2):
    """
    يكتشف ميل الرأس (alert / nodding)
    Input: Full face frame (224×224×3)

    استخدمنا EfficientNetB0 لأنه:
        - أدق من MobileNet في الـ spatial features
        - الـ head pose بيحتاج يفهم الوجه كله مش بس منطقة صغيرة
        - Compound scaling (width + depth + resolution معاً)
    """
    base = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = Input(shape=input_shape, name="head_input")
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomBrightness(0.15)(x)

    # EfficientNet بيعمل preprocessing داخلياً — مش محتاجين Lambda
    x = base(x, training=False)
    output = _classification_head(x, n_classes, dropout_rate=0.5, name_prefix="head")

    model = Model(inputs, output, name="HeadNoddingCNN")
    _compile_model(model, n_classes, lr=LEARNING_RATE)
    return model


# ──────────────────────────────────────────────────────────────
# ④  MicrosleepModel  —  CNN Feature Extractor + LSTM
# ──────────────────────────────────────────────────────────────
def build_microsleep_model(seq_len=30, input_shape=INPUT_SHAPE, n_classes=2):
    """
    يكتشف الـ microsleep (نوم لحظي < 3 ثوان)
    Input: Sequence من 30 frames (30, 224, 224, 3)

    Architecture:
        TimeDistributed(MobileNetV2) → يعالج كل frame على حدة
        → LSTM(128)                  → يفهم التسلسل الزمني
        → Dense head                 → binary output

    ليه CNN + LSTM؟
        - CNN: يستخرج features من كل frame
        - LSTM: يفهم إن السائق كان صاحي وبعدين بدأ ينام
        - Frame واحدة مش كفاية — محتاجين نشوف الـ pattern عبر الوقت
    """
    # Feature extractor مشترك
    feature_extractor = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        alpha=0.75           # أخف شوية عشان بنرن على 30 frames
    )
    feature_extractor.trainable = False

    # Input: (batch, seq_len, H, W, C)
    seq_input = Input(shape=(seq_len, *input_shape), name="sequence_input")

    # TimeDistributed: بيطبق نفس الـ CNN على كل frame في الـ sequence
    x = layers.TimeDistributed(
        layers.Lambda(lambda t: t * 2.0 - 1.0),
        name="preprocess"
    )(seq_input)

    x = layers.TimeDistributed(feature_extractor, name="cnn_backbone")(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D(), name="gap")(x)

    # الـ CNN طلع (batch, seq_len, features) — دلوقتي الـ LSTM
    x = layers.LSTM(128, return_sequences=True, dropout=0.3,
                    name="lstm1")(x)
    x = layers.LSTM(64, dropout=0.3, name="lstm2")(x)

    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    output = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(seq_input, output, name="MicrosleepModel")
    _compile_model(model, n_classes=2, lr=LEARNING_RATE)
    return model


# ──────────────────────────────────────────────────────────────
# HELPER: Compile Model
# ──────────────────────────────────────────────────────────────
def _compile_model(model, n_classes: int, lr: float):
    if n_classes == 2:
        loss = "binary_crossentropy"
        metrics = ["accuracy",
                   tf.keras.metrics.AUC(name="auc"),
                   tf.keras.metrics.Precision(name="precision"),
                   tf.keras.metrics.Recall(name="recall")]
    else:
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics
    )


# ──────────────────────────────────────────────────────────────
# FINE-TUNING  —  Unfreeze top layers بعد الـ initial training
# ──────────────────────────────────────────────────────────────
def unfreeze_top_layers(model, n_layers_to_unfreeze=30):
    """
    Phase 2 — Fine-tuning:
        بنـ unfreeze آخر N layers من الـ backbone
        ونعيد الـ training بـ learning rate أصغر بكتير

    ليه؟
        الـ early layers بتتعلم features عامة (edges, colors)
        الـ last layers بتتعلم features specific للمهمة
        → بنـ fine-tune الـ last layers على داتا الـ fatigue
    """
    # وصول للـ base model جوا الـ model
    for layer in model.layers:
        if hasattr(layer, 'layers'):   # هو الـ backbone
            backbone = layer
            break
    else:
        print("  ⚠ No nested backbone found — skipping unfreeze")
        return model

    backbone.trainable = True

    # Freeze كل حاجة إلا آخر N layers
    for layer in backbone.layers[:-n_layers_to_unfreeze]:
        layer.trainable = False

    trainable_count = sum(1 for l in backbone.layers if l.trainable)
    print(f"  Unfreezing: {trainable_count} / {len(backbone.layers)} backbone layers")

    # Re-compile بـ learning rate أصغر
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss=model.loss,
        metrics=model.metrics
    )
    return model
