import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json
import os
import matplotlib.pyplot as plt

# ============================================================
# FAST CONFIGURATION - Optimized for CPU
# ============================================================
IMG_SIZE = (160, 160)  # Smaller = faster
BATCH_SIZE = 64        # Larger batch for CPU efficiency
EPOCHS = 15            # Reduced epochs
LEARNING_RATE = 0.001  # Higher learning rate for faster convergence

print("="*60)
print("🚀 FAST TRAINING MODE - Optimized for CPU")
print("="*60)
print(f"\nConfiguration:")
print(f"  Image Size: {IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Max Epochs: {EPOCHS}")
print(f"  Expected Time: 2-3 hours on CPU")
print("="*60)

# ============================================================
# CREATE MODELS FOLDER
# ============================================================
os.makedirs('models', exist_ok=True)

# ============================================================
# DATA AUGMENTATION
# ============================================================
print("\n📊 Loading dataset...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    'skin_disease_dataset/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    'skin_disease_dataset/validation',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)

print(f"\n✓ Dataset loaded successfully!")
print(f"✓ Number of classes: {num_classes}")
print(f"✓ Training samples: {train_generator.samples}")
print(f"✓ Validation samples: {validation_generator.samples}")
print(f"\n📋 Classes:")
for i, (cls_name, idx) in enumerate(list(train_generator.class_indices.items())[:5]):
    print(f"   {idx}. {cls_name}")
if num_classes > 5:
    print(f"   ... and {num_classes - 5} more")

# ============================================================
# BUILD MODEL - MobileNetV2 (Faster than EfficientNet)
# ============================================================
print("\n🏗️ Building model...")

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Freeze base model
base_model.trainable = False

# Build complete model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

# Print model summary
print("\n✓ Model built successfully!")
print("\nModel Summary:")
model.summary()

print(f"\n📊 Total parameters: {model.count_params():,}")
trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
print(f"📊 Trainable parameters: {trainable:,}")

# ============================================================
# CALLBACKS
# ============================================================
callbacks = [
    # Save best model
    ModelCheckpoint(
        'models/skin_disease_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # Stop early if no improvement
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate on plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================
# TRAINING
# ============================================================
print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60)
print("\nThis will take approximately 2-3 hours...")
print("You can monitor progress below:\n")

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# SAVE CLASS NAMES
# ============================================================
print("\n💾 Saving class names...")
class_names = {v: k for k, v in train_generator.class_indices.items()}
with open('models/skin_class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)
print("✓ Class names saved to: models/skin_class_names.json")

# ============================================================
# EVALUATE MODEL
# ============================================================
print("\n" + "="*60)
print("📊 FINAL EVALUATION")
print("="*60)

# Evaluate on validation set
results = model.evaluate(validation_generator, verbose=0)
loss = results[0]
accuracy = results[1]
top3_accuracy = results[2]

print(f"\n✓ Validation Loss: {loss:.4f}")
print(f"✓ Validation Accuracy: {accuracy*100:.2f}%")
print(f"✓ Top-3 Accuracy: {top3_accuracy*100:.2f}%")

# ============================================================
# PLOT TRAINING HISTORY
# ============================================================
print("\n📈 Creating training history plots...")

def plot_history(history):
    """Plot training and validation accuracy/loss"""
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(14, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2)
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
    print("✓ Training history plot saved to: models/training_history.png")
    
    try:
        plt.show()
    except:
        pass

plot_history(history)

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)

print("\n📦 Saved Files:")
print("  ✓ models/skin_disease_model.h5")
print("  ✓ models/skin_class_names.json")
print("  ✓ models/training_history.png")

print(f"\n📊 Final Results:")
print(f"  ✓ Best Validation Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
print(f"  ✓ Best Top-3 Accuracy: {max(history.history['top_3_accuracy'])*100:.2f}%")
print(f"  ✓ Total Epochs Trained: {len(history.history['accuracy'])}")

print("\n🎯 Next Steps:")
print("  1. Verify model: python verify_model.py")
print("  2. Test model:   python test_skin_model.py")
print("  3. Run app:      python app.py")

print("\n" + "="*60)
print("Thank you for using Fast Training Mode! 🚀")
print("="*60)
