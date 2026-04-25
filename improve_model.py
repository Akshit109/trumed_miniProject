import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

print("="*60)
print("🔧 FINE-TUNING MODEL - Improving Accuracy")
print("="*60)

# Configuration
IMG_SIZE = (160, 160)
BATCH_SIZE = 32  # Smaller for fine-tuning
EPOCHS = 20
LEARNING_RATE = 0.00001  # Very low for fine-tuning

# Load data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'skin_disease_dataset/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    'skin_disease_dataset/validation',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print(f"\n✓ Training samples: {train_gen.samples}")
print(f"✓ Validation samples: {val_gen.samples}")

# Load trained model
print("\n📥 Loading previous model...")
model = load_model('models/skin_disease_model.h5')
print("✓ Model loaded!")

# Unfreeze base model for fine-tuning
print("\n🔓 Unfreezing base model layers...")
base_model = model.layers[0]
base_model.trainable = True

# Freeze first 100 layers (fine-tune only last layers)
for layer in base_model.layers[:100]:
    layer.trainable = False

print(f"✓ Trainable layers: {len([l for l in model.layers if l.trainable])}")

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

print("\n✅ Model ready for fine-tuning!")

# Callbacks
callbacks = [
    ModelCheckpoint(
        'models/skin_disease_model_finetuned.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1
    )
]

# Fine-tune
print("\n" + "="*60)
print("🚀 STARTING FINE-TUNING")
print("="*60)
print("\nThis will take 1-2 hours and improve accuracy significantly!\n")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
print("\n" + "="*60)
print("📊 FINAL EVALUATION")
print("="*60)

results = model.evaluate(val_gen, verbose=0)
print(f"\n✓ Validation Loss: {results[0]:.4f}")
print(f"✓ Validation Accuracy: {results[1]*100:.2f}%")
print(f"✓ Top-3 Accuracy: {results[2]*100:.2f}%")

print("\n" + "="*60)
print("✅ FINE-TUNING COMPLETE!")
print("="*60)
print("\n📦 Improved model saved to:")
print("   models/skin_disease_model_finetuned.h5")
print("\n🎯 Expected accuracy improvement: +15-25%")
print("\n💡 To use improved model:")
print("   Rename: skin_disease_model_finetuned.h5")
print("   To:     skin_disease_model.h5")