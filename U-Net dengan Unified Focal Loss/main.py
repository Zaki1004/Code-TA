from google.colab import drive
drive.mount('/content/drive')

# Import Library

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm

# Unified Focal Loss

class UnifiedFocalLoss(keras.losses.Loss):
    "Unified Focal Loss"

    def __init__(self,
                 num_classes=4,
                 delta=0.6,
                 gamma=0.9,
                 lambda_param=0.5,
                 asymmetric=True,
                 smooth=1e-6,
                 name="unified_focal_loss"):
        super().__init__(name=name)

        self.num_classes = num_classes
        self.delta = delta
        self.gamma = gamma
        self.lambda_param = lambda_param
        self.asymmetric = asymmetric
        self.smooth = smooth


    def modified_focal_loss(self, y_true, y_pred):
        "Modified Focal Loss: suppress background"
        y_pred = tf.clip_by_value(y_pred, self.smooth, 1.0 - self.smooth)
        bce = -y_true * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)

        if self.asymmetric:
            focal_weight_bg = self.delta * tf.pow(1.0 - p_t, 1.0 - self.gamma)
            focal_bg = focal_weight_bg * bce[:, :, :, 0:1]
            focal_rare = self.delta * bce[:, :, :, 1:]
            focal_loss = tf.concat([focal_bg, focal_rare], axis=-1)
        else:
            focal_weight = self.delta * tf.pow(1.0 - p_t, 1.0 - self.gamma)
            focal_loss = focal_weight * bce

        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    def modified_tversky_index(self, y_true, y_pred):
        "Modified Tversky Index"
        y_true_f = tf.reshape(y_true, [-1, self.num_classes])
        y_pred_f = tf.reshape(y_pred, [-1, self.num_classes])

        tp = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
        fp = tf.reduce_sum((1 - y_true_f) * y_pred_f, axis=0)
        fn = tf.reduce_sum(y_true_f * (1 - y_pred_f), axis=0)

        numerator = tp + self.smooth
        denominator = tp + self.delta * fp + (1.0 - self.delta) * fn + self.smooth

        return numerator / denominator

    def modified_focal_tversky_loss(self, y_true, y_pred):
        "Modified Focal Tversky Loss: enhance minorities"
        mti = self.modified_tversky_index(y_true, y_pred)

        if self.asymmetric:
            loss_bg = 1.0 - mti[0]
            loss_rare = tf.pow(1.0 - mti[1:], self.gamma)
            focal_tversky = tf.concat([[loss_bg], loss_rare], axis=0)
        else:
            focal_tversky = tf.pow(1.0 - mti, self.gamma)

        return tf.reduce_mean(focal_tversky)

    def call(self, y_true, y_pred):
        "Unified Loss"
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        focal_loss = self.modified_focal_loss(y_true, y_pred)
        focal_tversky_loss = self.modified_focal_tversky_loss(y_true, y_pred)

        return (self.lambda_param * focal_loss +
                (1.0 - self.lambda_param) * focal_tversky_loss)

# Data PreProcessing

class DataProcessor:
    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size

    def load_image(self, path):
        "Load, resize, normalize image ke [-1, 1]"
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        img = (img / 127.5) - 1.0
        return img.astype(np.float32)

    def load_mask(self, path):
        "Load mask RGB dengan mapping warna"
        mask_rgb = cv2.imread(path, cv2.IMREAD_COLOR)
        if mask_rgb is None:
            return None

        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.resize(mask_rgb, self.image_size, interpolation=cv2.INTER_NEAREST)

        color_map = {
            (0, 0, 0): 0,
            (255, 0, 0): 1,
            (0, 255, 0): 2, 
            (0, 0, 255): 3 
        }

        mask_labels = np.zeros(self.image_size, dtype=np.uint8)
        for color, label in color_map.items():
            match = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
            mask_labels[match] = label

        mask_onehot = np.zeros((*self.image_size, 4), dtype=np.float32)
        for i in range(4):
            mask_onehot[:, :, i] = (mask_labels == i).astype(np.float32)

        return mask_onehot

    def load_dataset(self, image_dir, mask_dir):
        "Load dataset berbagai format gambar"
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

        image_paths = sorted(image_paths)

        if not image_paths:
            print(f"!!! CRITICAL: NO IMAGE FILES in {image_dir}")
            return np.array([]), np.array([])

        print(f"Found {len(image_paths)} images")

        X, y = [], []
        skipped = 0

        for img_path in tqdm(image_paths, desc="Loading Data"):
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            mask_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                potential = os.path.join(mask_dir, base_name + ext)
                if os.path.exists(potential):
                    mask_path = potential
                    break

            if mask_path is None:
                skipped += 1
                continue

            try:
                image = self.load_image(img_path)
                mask = self.load_mask(mask_path)

                if image is None or mask is None:
                    skipped += 1
                    continue

                X.append(image)
                y.append(mask)
            except Exception as e:
                skipped += 1

        print(f"✓ Loaded: {len(X)} | ✗ Skipped: {skipped}")
        return np.array(X), np.array(y)

# Data Splitting

def split_data_70_15_15(X, y, train_ratio=0.7, val_ratio=0.15, random_state=42):
    "Split data 70/15/15"
    test_ratio = 1 - train_ratio - val_ratio

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), random_state=random_state
    )

    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_test_ratio), random_state=random_state
    )

    return X_train, y_train, X_val, y_val, X_test, y_test

# Arsitektur U-Net

def conv_block(inputs, filters, dropout_rate=0.2, l1_reg=1e-5):
    "Conv block dengan batch normal + dropout"
    x = layers.Conv2D(filters, 3, padding='same',
                      kernel_regularizer=regularizers.l1(l1_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(dropout_rate)(x)
    x = layers.Conv2D(filters, 3, padding='same',
                      kernel_regularizer=regularizers.l1(l1_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(dropout_rate)(x)
    return x

def encoder_block(inputs, filters, dropout_rate=0.2, l1_reg=1e-5):
    "Encoder: conv + pooling"
    x = conv_block(inputs, filters, dropout_rate, l1_reg)
    p = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x, p

def decoder_block(inputs, skip_connection, filters, dropout_rate=0.2, l1_reg=1e-5):
    "Decoder: transpose conv + concatenate + conv"
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same',
                               kernel_regularizer=regularizers.l1(l1_reg))(inputs)
    x = layers.concatenate([x, skip_connection])
    x = conv_block(x, filters, dropout_rate, l1_reg)
    return x

def build_unet(input_shape=(256, 256, 3), num_classes=4, dropout_rate=0.2, l1_reg=1e-5):
    "Build U-Net"
    inputs = layers.Input(shape=input_shape)

    s1, p1 = encoder_block(inputs, 64, dropout_rate, l1_reg)
    s2, p2 = encoder_block(p1, 128, dropout_rate, l1_reg)
    s3, p3 = encoder_block(p2, 256, dropout_rate, l1_reg)
    s4, p4 = encoder_block(p3, 512, dropout_rate, l1_reg)
    s5, p5 = encoder_block(p4, 1024, dropout_rate, l1_reg)

    b = conv_block(p5, 2048, dropout_rate, l1_reg)

    d1 = decoder_block(b, s5, 1024, dropout_rate, l1_reg)
    d2 = decoder_block(d1, s4, 512, dropout_rate, l1_reg)
    d3 = decoder_block(d2, s3, 256, dropout_rate, l1_reg)
    d4 = decoder_block(d3, s2, 128, dropout_rate, l1_reg)
    d5 = decoder_block(d4, s1, 64, dropout_rate, l1_reg)

    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(d5)

    model = models.Model(inputs=inputs, outputs=outputs, name="U-Net")
    return model

# Metrics Evaluasi

def dice_background(y_true, y_pred, smooth=1e-6):
    "Dice untuk Background"
    y_true_f = tf.reshape(y_true[:, :, :, 0], [-1])
    y_pred_f = tf.reshape(y_pred[:, :, :, 0], [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_lubang(y_true, y_pred, smooth=1e-6):
    "Dice untuk Lubang"
    y_true_f = tf.reshape(y_true[:, :, :, 1], [-1])
    y_pred_f = tf.reshape(y_pred[:, :, :, 1], [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_retakan(y_true, y_pred, smooth=1e-6):
    "Dice untuk Retakan"
    y_true_f = tf.reshape(y_true[:, :, :, 2], [-1])
    y_pred_f = tf.reshape(y_pred[:, :, :, 2], [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_rutting(y_true, y_pred, smooth=1e-6):
    "Dice untuk Rutting"
    y_true_f = tf.reshape(y_true[:, :, :, 3], [-1])
    y_pred_f = tf.reshape(y_pred[:, :, :, 3], [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def mean_damage_dice(y_true, y_pred):
    "Mean Dice damage classes"
    return (dice_lubang(y_true, y_pred) +
            dice_retakan(y_true, y_pred) +
            dice_rutting(y_true, y_pred)) / 3.0

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    "Dice coefficient overall"
    y_true_f = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
    y_pred_f = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f + y_pred_f, axis=0)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)

def iou_metric(y_true, y_pred, smooth=1e-6):
    "IoU metric"
    y_true_f = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
    y_pred_f = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f + y_pred_f, axis=0) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)

# Inisiasi Training

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS_TOTAL = 150
LEARNING_RATE = 3e-4 
DROPOUT_RATE = 0.2
L1_REG = 1e-5

ALL_IMAGE_DIR = "/content/drive/MyDrive/New/AllDataImagesCleaning"
ALL_MASK_DIR = "/content/drive/MyDrive/New/AllDataMasksCleaning"
OUTPUT_DIR = "/content/drive/MyDrive/New/Gamma-0.9-UnifiedFocalLoss-E150"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "unet_best_ufl.keras")
HISTORY_SAVE_PATH = os.path.join(OUTPUT_DIR, "training_history_ufl.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Dataset

data_processor = DataProcessor(IMAGE_SIZE)
X_all, y_all = data_processor.load_dataset(ALL_IMAGE_DIR, ALL_MASK_DIR)

if len(X_all) == 0:
    raise ValueError("Tidak ada data yang dimuat!")

# Split 70/15/15
X_train, y_train, X_val, y_val, X_test, y_test = split_data_70_15_15(X_all, y_all)

print(f"\nTraining: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Testing: {len(X_test)}")

# Build Model

print("\n" + "="*70)
print("Build Model dengan Unified Focal Loss")
print("="*70)

model = build_unet(
    input_shape=(*IMAGE_SIZE, 3),
    num_classes=4,
    dropout_rate=DROPOUT_RATE,
    l1_reg=L1_REG
)

if os.path.exists(MODEL_SAVE_PATH):
    try:
        model.load_weights(MODEL_SAVE_PATH)
        print(f"✓ Loaded existing weights from {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"⚠ Failed to load weights: {e}")

model.summary()

GAMMA_VALUE = 0.9

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=UnifiedFocalLoss(
        num_classes=4,
        delta=0.6,          
        gamma=GAMMA_VALUE,  
        lambda_param=0.5,   
        asymmetric=True    
    ),
    metrics=[
        'accuracy',
        dice_coefficient,
        iou_metric,
        dice_background,
        dice_lubang,
        dice_retakan,      
        dice_rutting,      
        mean_damage_dice   
    ]
)

# Tuning Hyperparameter ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

callbacks = [
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_mean_damage_dice',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_mean_damage_dice',
        mode='max',
        patience=30,
        verbose=1,
        restore_best_weights=True,
        min_delta=0.001
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-7,
        verbose=1,
        cooldown=5
    )
]

# Training Model

history_lama = None
start_epoch = 0

if os.path.exists(HISTORY_SAVE_PATH):
    try:
        with open(HISTORY_SAVE_PATH, 'r') as f:
            history_dict = json.load(f)
        history_lama = tf.keras.callbacks.History()
        history_lama.history = history_dict
        start_epoch = len(history_lama.history['loss'])
        print(f"Resuming from Epoch {start_epoch + 1}")
    except Exception as e:
        print(f"Could not load history: {e}")

if start_epoch >= EPOCHS_TOTAL:
    print(f"Training completed ({start_epoch} epochs)")
    history_gabungan = history_lama
else:
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING (Epoch {start_epoch+1} to {EPOCHS_TOTAL})")
    print(f"{'='*70}\n")

    history_lanjutan = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_TOTAL,
        initial_epoch=start_epoch,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    history_gabungan = combine_histories(history_lama, history_lanjutan)

    with open(HISTORY_SAVE_PATH, 'w') as f:
        json.dump(history_gabungan.history, f)
    print(f"\n✓ History saved to {HISTORY_SAVE_PATH}")

print("\n" + "="*70)
print("TRAINING COMPLETED!")
print("="*70)

# Grafik Training

def plot_training_history(history_gabungan):
    "Plot training curves"
    if history_gabungan is None or not history_gabungan.history:
        print("Plotting dibatalkan karena tidak ada riwayat training yang valid.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    epochs = range(1, len(history_gabungan.history['loss']) + 1)

    # F1 Score
    axes[0, 0].plot(epochs, history_gabungan.history['dice_coefficient'], label='Training F1', linewidth=2)
    axes[0, 0].plot(epochs, history_gabungan.history['val_dice_coefficient'], label='Validation F1', linewidth=2)
    axes[0, 0].set_title('F1 Score (Dice Coefficient)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('F1 Score'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # IoU
    axes[0, 1].plot(epochs, history_gabungan.history['iou_metric'], label='Training IoU', linewidth=2)
    axes[0, 1].plot(epochs, history_gabungan.history['val_iou_metric'], label='Validation IoU', linewidth=2)
    axes[0, 1].set_title('Mean IoU', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('IoU'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # Loss
    axes[1, 0].plot(epochs, history_gabungan.history['loss'], label='Training Loss', linewidth=2)
    axes[1, 0].plot(epochs, history_gabungan.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1, 0].set_title('Unified Focal Loss', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Loss'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[1, 1].plot(epochs, history_gabungan.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1, 1].plot(epochs, history_gabungan.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Accuracy'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_256.png'), dpi=300, bbox_inches='tight')
    plt.show()

plot_training_history(history_gabungan)

# Metrils Evaluasi per Class

def evaluate_per_class(model, X_data, y_data):
    "Evaluasi per class pada set data tertentu"
    if len(X_data) == 0:
        print("Evaluation dibatalkan karena data pengujian kosong.")
        return

    if os.path.exists(MODEL_SAVE_PATH):
        try:
            final_model = build_unet(
                input_shape=(*IMAGE_SIZE, 3),
                num_classes=4,
                dropout_rate=DROPOUT_RATE,
                l1_reg=L1_REG)

            final_model.load_weights(MODEL_SAVE_PATH)
            print("Loaded best weights for final evaluation.")
            model_to_evaluate = final_model
        except Exception as e:
            print(f"WARNING: Could not load best weights. Using current model state. Error: {e}")
            model_to_evaluate = model
    else:
        model_to_evaluate = model
        print("WARNING: Best model weights not found. Using current model state for evaluation.")

    y_pred = model_to_evaluate.predict(X_data, batch_size=4)

    class_names = ['Background', 'Pothole', 'Crack', 'Rutting']
    print("\n" + "="*60)
    print("CLASS-SPECIFIC EVALUATION METRICS")
    print("="*60)

    for i, class_name in enumerate(class_names):
        y_true_class = y_data[:, :, :, i].flatten()
        y_pred_class = y_pred[:, :, :, i].flatten()

        intersection = np.sum(y_true_class * y_pred_class)
        union = np.sum(y_true_class + y_pred_class)

        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        iou = (intersection + 1e-6) / (union - intersection + 1e-6)

        print(f"\n{class_name}:")
        print(f"  F1 Score: {dice:.4f}")
        print(f"  IoU:      {iou:.4f}")

    y_pred_tensor = tf.convert_to_tensor(y_pred)
    y_data_tensor = tf.convert_to_tensor(y_data)
    mean_dice = dice_coefficient(y_data, y_pred_tensor).numpy()
    mean_iou = iou_metric(y_data, y_pred_tensor).numpy()

    print(f"\n{'='*60}")
    print(f"OVERALL METRICS")
    print(f"{'='*60}")
    print(f"Mean F1 Score: {mean_dice:.4f}")
    print(f"Mean IoU:      {mean_iou:.4f}")
    print(f"{'='*60}\n")

print("\nEvaluating model on FINAL TEST set...")
evaluate_per_class(model, X_test, y_test)

# Confusion Matrix

def plot_confusion_matrix(model, X_data, y_data, class_names, output_path, title="Confusion Matrix"):
    if len(X_data) == 0:
        print("No data for confusion matrix")
        return

    print("\nGenerating confusion matrix...")
    y_pred = model.predict(X_data, batch_size=4, verbose=1)

    y_true_labels = np.argmax(y_data, axis=-1).flatten()
    y_pred_labels = np.argmax(y_pred, axis=-1).flatten()

    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=[0, 1, 2, 3])

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Pixels'})

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted label', fontsize=12, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {output_path}")
    plt.show()

    print("\n" + "="*60)
    print("CONFUSION MATRIX STATISTICS")
    print("="*60)
    for i, name in enumerate(class_names):
        total_true = np.sum(cm[i, :])
        total_pred = np.sum(cm[:, i])
        correct = cm[i, i]

        precision = correct / total_pred if total_pred > 0 else 0
        recall = correct / total_true if total_true > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n{name}:")
        print(f"  TP:        {correct:,}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")
    print("="*60)

class_names = ['Background', 'Pothole', 'Crack', 'Rutting']

if os.path.exists(MODEL_SAVE_PATH):
    try:
        best_model = build_unet((*IMAGE_SIZE, 3), 4, DROPOUT_RATE, L1_REG)
        best_model.load_weights(MODEL_SAVE_PATH)
        model_for_cm = best_model
    except:
        model_for_cm = model
else:
    model_for_cm = model

print("\n" + "="*60)
print("GENERATING CONFUSION MATRIX")
print("="*60)
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_test_256.png')
plot_confusion_matrix(model_for_cm, X_test, y_test, class_names, cm_path, "Confusion Matrix")

# Hasil Visualisasi

def visualize_predictions(model, X_data, y_data, num_samples=5, title_prefix=""):
    "Visualisasi prediksi"
    if len(X_data) == 0:
        print("Visualisasi dibatalkan karena data kosong.")
        return

    if os.path.exists(MODEL_SAVE_PATH):
        try:
            final_model = build_unet(
                input_shape=(*IMAGE_SIZE, 3),
                num_classes=4,
                dropout_rate=DROPOUT_RATE,
                l1_reg=L1_REG)
            final_model.load_weights(MODEL_SAVE_PATH)
            model_to_predict = final_model
        except:
            model_to_predict = model
    else:
        model_to_predict = model

    indices = np.random.choice(len(X_data), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))

    color_map = {
        1: [255, 0, 0],      
        2: [0, 255, 0],      
        3: [0, 0, 255]       
    }

    for idx, i in enumerate(indices):
        # Input Citra
        img = (X_data[i] + 1.0) / 2.0  
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f'{title_prefix} Original Image')
        axes[idx, 0].axis('off')

        # Ground truth
        gt_mask = np.argmax(y_data[i], axis=-1)
        gt_colored = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            gt_colored[gt_mask == class_id] = color
        axes[idx, 1].imshow(gt_colored)
        axes[idx, 1].set_title('Ground Truth')
        axes[idx, 1].axis('off')

        # Hasil Prediksi
        pred = model_to_predict.predict(X_data[i:i+1], verbose=0)[0]
        pred_mask = np.argmax(pred, axis=-1)
        pred_colored = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            pred_colored[pred_mask == class_id] = color
        axes[idx, 2].imshow(pred_colored)
        axes[idx, 2].set_title('Prediction')
        axes[idx, 2].axis('off')

    plt.tight_layout()
   plt.suptitle(f'Visualizing Predictions on Test Data', y=1.02, fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, f'predictions_{title_prefix.lower()}.png'), dpi=300, bbox_inches='tight')
    plt.show()

print("\nVisualizing predictions on TEST set...")
visualize_predictions(model, X_test, y_test, num_samples=5, title_prefix="TEST")

print("\n" + "="*60)
print("TRAINING COMPLETED & FINAL EVALUATION ON TEST SET DONE!")
print("="*60)