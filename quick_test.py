import tensorflow as tf
from PIL import Image
import numpy as np
import json
import glob
import os

print("="*60)
print("🧪 QUICK TEST - Random Validation Images")
print("="*60)

# Load model
print("\n📥 Loading model...")
try:
    model = tf.keras.models.load_model('models/skin_disease_model.h5')
    with open('models/skin_class_names.json', 'r') as f:
        classes = json.load(f)
    print("✅ Model loaded!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n🔧 Fix:")
    print("   copy models\\skin_disease_model_finetuned.h5 models\\skin_disease_model.h5")
    exit(1)

# Get random test images (5 from different diseases)
print("\n🔍 Finding test images...")
test_images = glob.glob('skin_disease_dataset/validation/*/*')[:5]

if not test_images:
    test_images = glob.glob('skin_disease_dataset\\validation\\*\\*')[:5]

if not test_images:
    print("❌ No test images found!")
    print("   Make sure dataset exists: skin_disease_dataset/validation/")
    exit(1)

print(f"✅ Found {len(test_images)} test images\n")
print("="*60)

# Test each image
correct = 0
total = 0

for img_path in test_images:
    try:
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((160, 160))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        
        # Top 3 predictions
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        
        # Get actual disease from folder path
        path_parts = img_path.replace('\\', '/').split('/')
        actual_disease = path_parts[-2]
        filename = path_parts[-1]
        
        # Get predicted disease
        predicted_disease = classes[str(top3_indices[0])]
        confidence = predictions[0][top3_indices[0]] * 100
        
        # Check if correct
        is_correct = (actual_disease == predicted_disease)
        if is_correct:
            correct += 1
        total += 1
        
        # Display result
        print(f"\n📸 Image {total}: {filename}")
        print(f"   Actual:    {actual_disease}")
        print(f"   Predicted: {predicted_disease}")
        print(f"   Confidence: {confidence:.2f}%")
        print(f"   Status: {'✅ CORRECT' if is_correct else '❌ WRONG'}")
        
        # Show top 3
        print(f"\n   Top 3 predictions:")
        for i, idx in enumerate(top3_indices, 1):
            disease = classes[str(idx)]
            conf = predictions[0][idx] * 100
            # Truncate long names
            if len(disease) > 50:
                disease = disease[:47] + "..."
            print(f"   {i}. {disease} ({conf:.2f}%)")
        
        print("-" * 60)
        
    except Exception as e:
        print(f"\n❌ Error processing {img_path}: {e}")

# Summary
print("\n" + "="*60)
print("📊 TEST SUMMARY")
print("="*60)
print(f"\n   Total images tested: {total}")
print(f"   Correct predictions: {correct}")
print(f"   Wrong predictions: {total - correct}")

if total > 0:
    accuracy = (correct / total) * 100
    print(f"   Accuracy: {accuracy:.2f}%")
    
    if accuracy >= 70:
        print(f"\n   ✅ Excellent! Model is performing well!")
    elif accuracy >= 50:
        print(f"\n   ✅ Good! Model is working properly!")
    elif accuracy >= 30:
        print(f"\n   ⚠️  Fair - Model works but could be better")
    else:
        print(f"\n   ⚠️  Low accuracy - Consider more training")

print("\n💡 Note: This is a small sample (5 images)")
print("   For full evaluation, use validation set (4000+ images)")

print("\n🎯 Next steps:")
print("   1. Test with your own image: python test_skin_model.py")
print("   2. Run Flask app: python app.py")

print("\n" + "="*60)
