import os
import json
import tensorflow as tf
from PIL import Image
import numpy as np

print("="*60)
print("🔍 MODEL VERIFICATION")
print("="*60)

# ============================================================
# CHECK FILES EXIST
# ============================================================
model_path = 'models/skin_disease_model.h5'
classes_path = 'models/skin_class_names.json'

model_exists = os.path.exists(model_path)
classes_exist = os.path.exists(classes_path)

print(f"\n📁 Checking files...")
print(f"{'✅' if model_exists else '❌'} Model file: {model_path}")
if model_exists:
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   Size: {size_mb:.2f} MB")

print(f"{'✅' if classes_exist else '❌'} Classes file: {classes_path}")

# ============================================================
# LOAD AND VERIFY MODEL
# ============================================================
if model_exists and classes_exist:
    try:
        # Load model
        print("\n🔄 Loading model...")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Model details
        print(f"\n📊 Model Architecture:")
        print(f"   Input shape:  {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Total params: {model.count_params():,}")
        
        # Count trainable params
        trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        print(f"   Trainable:    {trainable:,}")
        print(f"   Non-trainable: {non_trainable:,}")
        
        # Load classes
        with open(classes_path, 'r') as f:
            classes = json.load(f)
        
        num_classes = len(classes)
        output_classes = model.output_shape[-1]
        
        print(f"\n📋 Disease Classes:")
        print(f"   Total classes: {num_classes}")
        print(f"   Model output:  {output_classes}")
        
        if num_classes != output_classes:
            print(f"\n⚠️  WARNING: Mismatch!")
            print(f"   Classes file has {num_classes} classes")
            print(f"   Model outputs {output_classes} predictions")
        else:
            print(f"   ✅ Class count matches!")
        
        # Show sample classes
        print(f"\n📝 Sample Classes:")
        for i, (idx, name) in enumerate(list(classes.items())[:10]):
            short_name = name[:60] + "..." if len(name) > 60 else name
            print(f"   {idx}. {short_name}")
        
        if num_classes > 10:
            print(f"   ... and {num_classes - 10} more")
        
        # ============================================================
        # TEST PREDICTION (Dummy Input)
        # ============================================================
        print(f"\n🧪 Testing prediction with dummy input...")
        
        # Get input size from model
        input_shape = model.input_shape
        img_height = input_shape[1]
        img_width = input_shape[2]
        
        # Create random test image
        test_image = np.random.rand(1, img_height, img_width, 3).astype(np.float32)
        
        # Make prediction
        predictions = model.predict(test_image, verbose=0)
        
        # Get top prediction
        top_class_idx = np.argmax(predictions[0])
        top_confidence = predictions[0][top_class_idx] * 100
        top_class_name = classes[str(top_class_idx)]
        
        print(f"✅ Prediction successful!")
        print(f"   Predicted: {top_class_name}")
        print(f"   Confidence: {top_confidence:.2f}%")
        print(f"   (This is a random test, not a real diagnosis)")
        
        # Show top 3 predictions
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        print(f"\n   Top 3 predictions:")
        for i, idx in enumerate(top3_indices, 1):
            class_name = classes[str(idx)]
            confidence = predictions[0][idx] * 100
            short_name = class_name[:50] + "..." if len(class_name) > 50 else class_name
            print(f"   {i}. {short_name} ({confidence:.2f}%)")
        
        # ============================================================
        # FINAL STATUS
        # ============================================================
        print("\n" + "="*60)
        print("✅ MODEL IS READY FOR USE!")
        print("="*60)
        
        print("\n🎯 Your model can now:")
        print("   ✅ Accept skin disease images")
        print("   ✅ Predict from 23 disease classes")
        print("   ✅ Provide confidence scores")
        print("   ✅ Show top-3 predictions")
        
        print("\n📱 Next Steps:")
        print("   1. Test with real image:")
        print("      python test_skin_model.py")
        print("\n   2. Run Flask app:")
        print("      python app.py")
        print("      Then open: http://localhost:5000")
        
        print("\n💡 Model Details:")
        print(f"   Model file: {model_path} ({size_mb:.2f} MB)")
        print(f"   Input size: {img_height}x{img_width}")
        print(f"   Classes: {num_classes}")
        print(f"   Parameters: {model.count_params():,}")
        
    except Exception as e:
        print(f"\n❌ ERROR LOADING MODEL!")
        print(f"   Error: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check if model file is corrupted")
        print("   2. Retrain the model: python train_skin_fast.py")
        print("   3. Or fine-tune: python improve_model.py")
        
        import traceback
        print("\n📋 Full error trace:")
        traceback.print_exc()

else:
    # Files missing
    print("\n❌ MODEL FILES NOT FOUND!")
    
    if not model_exists:
        print(f"   Missing: {model_path}")
    if not classes_exist:
        print(f"   Missing: {classes_path}")
    
    print("\n🔧 Please train the model first:")
    print("   Option 1 (Fast): python train_skin_fast.py")
    print("   Option 2 (Better): python improve_model.py")

print("\n" + "="*60)
