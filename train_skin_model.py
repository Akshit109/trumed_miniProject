import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIGURATION
# ============================================================

# Update these paths to your dataset
DATASET_PATH = 'skin_disease_dataset'  # Your dataset folder
MODEL_SAVE_PATH = 'models/skin_disease_model.h5'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

# Dataset structure should be:
# skin_disease_dataset/
#   ├── train/
#   │   ├── Acne/
#   │   ├── Eczema/
#   │   ├── Melanoma/
#   │   └── ...
#   └── validation/
#       ├── Acne/
#       ├── Eczema/
#       └── ...

# ============================================================
# DATA AUGMENTATION (Increases accuracy by 10-15%)
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    f'{DATASET_PATH}/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    f'{DATASET_PATH}/validation',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"\n✓ Found {num_classes} skin disease classes")
print(f"✓ Training samples: {train_generator.samples}")
print(f"✓ Validation samples: {validation_generator.samples}")
print(f"\nClasses: {list(train_generator.class_indices.keys())}\n")

# ============================================================
# MODEL ARCHITECTURE - EfficientNetB3 (State-of-the-art)
# ============================================================

def create_model(num_classes):
    """
    EfficientNetB3 with transfer learning
    Expected accuracy: 85-95% (with proper dataset)
    """
    # Load pre-trained EfficientNetB3
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    model = models.Sequential([
        base_model,
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers with dropout
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

# ============================================================
# ALTERNATIVE MODEL - ResNet50V2 (Also very good)
# ============================================================

def create_resnet_model(num_classes):
    """
    ResNet50V2 with transfer learning
    Expected accuracy: 83-92%
    """
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

# ============================================================
# CUSTOM CNN MODEL (If you don't have pre-trained weights)
# ============================================================

def create_custom_cnn(num_classes):
    """
    Custom CNN from scratch
    Expected accuracy: 70-85% (requires more data)
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                     input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, None

# ============================================================
# CREATE AND COMPILE MODEL
# ============================================================

print("Creating model...")
model, base_model = create_model(num_classes)  # Using EfficientNetB3

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

model.summary()

# ============================================================
# CALLBACKS FOR BETTER TRAINING
# ============================================================

callbacks = [
    # Save best model
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # Stop if no improvement
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate if plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================
# TRAINING - PHASE 1 (Frozen base model)
# ============================================================

print("\n" + "="*60)
print("PHASE 1: Training with frozen base model")
print("="*60 + "\n")

history1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# TRAINING - PHASE 2 (Fine-tuning)
# ============================================================

if base_model is not None:
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning the model")
    print("="*60 + "\n")
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze the first 80% of layers
    for layer in base_model.layers[:int(len(base_model.layers) * 0.8)]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    # Continue training
    history2 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        initial_epoch=15,
        callbacks=callbacks,
        verbose=1
    )

# ============================================================
# SAVE MODEL AND CLASS NAMES
# ============================================================

print(f"\n✓ Model saved to: {MODEL_SAVE_PATH}")

# Save class names
import json
class_names = {v: k for k, v in train_generator.class_indices.items()}
with open('models/skin_class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)
print("✓ Class names saved to: models/skin_class_names.json")

# ============================================================
# EVALUATE MODEL
# ============================================================

print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60 + "\n")

loss, accuracy, top3_accuracy = model.evaluate(validation_generator)
print(f"\n✓ Validation Loss: {loss:.4f}")
print(f"✓ Validation Accuracy: {accuracy*100:.2f}%")
print(f"✓ Top-3 Accuracy: {top3_accuracy*100:.2f}%")

# ============================================================
# PLOT TRAINING HISTORY
# ============================================================

def plot_history(history1, history2=None):
    """Plot training history"""
    
    if history2 is not None:
        # Combine histories
        acc = history1.history['accuracy'] + history2.history['accuracy']
        val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
        loss = history1.history['loss'] + history2.history['loss']
        val_loss = history1.history['val_loss'] + history2.history['val_loss']
    else:
        acc = history1.history['accuracy']
        val_acc = history1.history['val_accuracy']
        loss = history1.history['loss']
        val_loss = history1.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    print("\n✓ Training history plot saved to: models/training_history.png")
    plt.show()

if base_model is not None:
    plot_history(history1, history2)
else:
    plot_history(history1)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)