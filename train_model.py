import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


# =====================================
# SETTINGS
# =====================================

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30
SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "face_spoof_model.keras")

LCC_DATASET_DIR = os.path.join(BASE_DIR, "dataset_lcc")
WEBCAM_DATASET_DIR = os.path.join(BASE_DIR, "dataset_webcam")

STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

ACCURACY_GRAPH_PATH = os.path.join(STATIC_DIR, "accuracy_graph.png")
CONFUSION_MATRIX_PATH = os.path.join(STATIC_DIR, "confusion_matrix.png")

DATASET_DIR = os.environ.get(
    "FACE_SPOOF_DATASET",
    WEBCAM_DATASET_DIR if os.path.isdir(WEBCAM_DATASET_DIR)
    else LCC_DATASET_DIR
)

tf.random.set_seed(SEED)
np.random.seed(SEED)


# =====================================
# CHECK DATASET
# =====================================

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

for split in ["train", "validation", "test"]:
    split_path = os.path.join(DATASET_DIR, split)

    if not os.path.isdir(split_path):
        raise FileNotFoundError(
            f"Missing dataset folder: {split_path}"
        )

print("\nSelected Dataset:", DATASET_DIR)


# =====================================
# LOAD TRAIN DATASET
# =====================================

print("\nLoading Training Dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["real", "spoof"],
    seed=SEED,
    shuffle=True
)

print("\nTRAIN DATASET CLASS NAMES:")
print(train_ds.class_names)


# =====================================
# LOAD VALIDATION DATASET
# =====================================

print("\nLoading Validation Dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "validation"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["real", "spoof"],
    shuffle=False
)


# =====================================
# LOAD TEST DATASET
# =====================================

print("\nLoading Test Dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["real", "spoof"],
    shuffle=False
)

print("\nCLASS LABELS")
print("REAL  = 0")
print("SPOOF = 1")


# =====================================
# DATA OPTIMIZATION
# =====================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# =====================================
# CLASS WEIGHTS
# =====================================

train_class_counts = {}

for class_name in ["real", "spoof"]:

    folder = os.path.join(
        DATASET_DIR,
        "train",
        class_name
    )

    train_class_counts[class_name] = len([
        filename
        for filename in os.listdir(folder)
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ])

if train_class_counts["real"] == 0 or train_class_counts["spoof"] == 0:
    raise ValueError("Training dataset must contain REAL and SPOOF images.")

train_total = sum(train_class_counts.values())

class_weights = {
    class_id: train_total / (2 * train_class_counts[class_name])
    for class_id, class_name in enumerate(["real", "spoof"])
}

print("\nTraining Class Counts:", train_class_counts)
print("Training Class Weights:", class_weights)


# =====================================
# DATA AUGMENTATION
# =====================================

data_augmentation = models.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.03),

    layers.RandomZoom(0.08)

])


# =====================================
# BUILD CNN MODEL
# =====================================

model = models.Sequential([

    layers.Input(shape=(128, 128, 3)),

    data_augmentation,

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(
        32, 3,
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(
        64, 3,
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(
        128, 3,
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(
        256, 3,
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.4),

    layers.Dense(1, activation="sigmoid")

])


# =====================================
# COMPILE MODEL
# =====================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=["accuracy"]

)

print("\nMODEL SUMMARY")
model.summary()


# =====================================
# CALLBACKS
# =====================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.000001,
    verbose=1
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# =====================================
# TRAIN MODEL
# =====================================

print("\n================================")
print("MODEL TRAINING STARTED")
print("================================")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    class_weight=class_weights,
    epochs=EPOCHS,
    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ]
)


# =====================================
# SAVE MODEL ACCURACY GRAPH
# =====================================

print("\nSaving Accuracy Graph...")

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Face Spoof Detection - Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    ACCURACY_GRAPH_PATH,
    dpi=150
)

plt.close()

print("Accuracy Graph Saved:", ACCURACY_GRAPH_PATH)


# =====================================
# LOAD BEST MODEL
# =====================================

print("\nLoading Best Model...")

best_model = tf.keras.models.load_model(MODEL_PATH)


# =====================================
# TEST MODEL
# =====================================

print("\n================================")
print("TESTING MODEL")
print("================================")

test_loss, test_accuracy = best_model.evaluate(test_ds)

print("\nFINAL TEST RESULTS")
print("----------------------------")

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")


# =====================================
# PREDICTIONS
# =====================================

print("\nGenerating Predictions...")

y_true = np.concatenate([
    labels.numpy().flatten()
    for images, labels in test_ds
]).astype(int)

y_scores = best_model.predict(test_ds, verbose=1).flatten()

y_pred = (y_scores >= 0.5).astype(int)


# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

print("\nCONFUSION MATRIX")
print("----------------------------")
print(cm)


# =====================================
# SAVE CONFUSION MATRIX GRAPH
# =====================================

print("\nSaving Confusion Matrix...")

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["REAL", "SPOOF"]
)

fig, ax = plt.subplots(figsize=(6, 5))

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title("Face Spoof Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=150
)

plt.close()

print("Confusion Matrix Saved:", CONFUSION_MATRIX_PATH)


# =====================================
# CLASSIFICATION REPORT
# =====================================

print("\nCLASSIFICATION REPORT")
print("----------------------------")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["REAL", "SPOOF"],
        zero_division=0
    )
)


# =====================================
# SAVE MODEL
# =====================================

best_model.save(MODEL_PATH)


# =====================================
# FINAL OUTPUT
# =====================================

print("\n================================")
print("MODEL TRAINING COMPLETED!")
print("================================")

print("Model Saved:", MODEL_PATH)
print("Accuracy Graph:", ACCURACY_GRAPH_PATH)
print("Confusion Matrix:", CONFUSION_MATRIX_PATH)

print("\nREAL = 0")
print("SPOOF = 1")