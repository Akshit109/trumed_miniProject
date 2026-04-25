import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

print("="*60)
print("🧪 TESTING SKIN DISEASE MODEL")
print("="*60)

# ============================================================
# LOAD MODEL AND CLASSES
# ============================================================
print("\n📥 Loading model...")

try:
    model = tf.keras.models.load_model('models/skin_disease_model.h5')
    print("✅ Model loaded!")
    
    with open('models/skin_class_names.json', 'r') as f:
        disease_classes = json.load(f)
    print(f"✅ Loaded {len(disease_classes)} disease classes")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\n🔧 Please verify model first:")
    print("   1. copy models\\skin_disease_model_finetuned.h5 models\\skin_disease_model.h5")
    print("   2. python verify_model.py")
    exit(1)

# Get input size from model
input_shape = model.input_shape
img_height = input_shape[1]
img_width = input_shape[2]

print(f"✅ Model expects {img_height}x{img_width} images")

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_skin_disease(image_path):
    """Predict skin disease from image"""
    
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        
        img = img.resize((img_width, img_height))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Get top 5 predictions
        top5_indices = np.argsort(predictions[0])[-5:][::-1]
        
        results = []
        for idx in top5_indices:
            disease = disease_classes[str(idx)]
            confidence = predictions[0][idx] * 100
            results.append({
                'disease': disease,
                'confidence': confidence
            })
        
        return {
            'success': True,
            'results': results,
            'original_size': original_size
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# ============================================================
# GET RISK LEVEL
# ============================================================
def get_risk_level(disease_name, confidence):
    """Determine risk level based on disease name"""
    
    high_risk_keywords = [
        'cancer', 'carcinoma', 'melanoma', 'malignant', 
        'lesions', 'tumors'
    ]
    
    medium_risk_keywords = [
        'infection', 'bacterial', 'fungal', 'viral',
        'cellulitis', 'impetigo'
    ]
    
    disease_lower = disease_name.lower()
    
    # Check for high risk
    for keyword in high_risk_keywords:
        if keyword in disease_lower:
            return "🔴 HIGH RISK - Seek immediate medical attention"
    
    # Check for medium risk
    for keyword in medium_risk_keywords:
        if keyword in disease_lower:
            return "🟡 MEDIUM RISK - Consult a doctor soon"
    
    # Low confidence = uncertain
    if confidence < 40:
        return "⚪ UNCERTAIN - Please consult a dermatologist"
    
    return "🟢 LOW RISK - Monitor and consult if worsens"

# ============================================================
# GET RECOMMENDATIONS
# ============================================================
def get_recommendations(disease_name):
    """Get basic recommendations based on disease"""
    
    disease_lower = disease_name.lower()
    
    if 'acne' in disease_lower or 'rosacea' in disease_lower:
        return [
            "Keep skin clean with gentle cleanser",
            "Avoid touching or picking affected areas",
            "Consider over-the-counter acne treatments",
            "Consult dermatologist for persistent cases"
        ]
    
    elif 'cancer' in disease_lower or 'melanoma' in disease_lower or 'carcinoma' in disease_lower:
        return [
            "⚠️ URGENT: Schedule appointment with dermatologist immediately",
            "Avoid sun exposure and use high SPF sunscreen",
            "Do not attempt self-treatment",
            "Get professional biopsy if recommended"
        ]
    
    elif 'eczema' in disease_lower or 'dermatitis' in disease_lower:
        return [
            "Moisturize regularly with fragrance-free products",
            "Avoid known triggers and irritants",
            "Use gentle, hypoallergenic soaps",
            "Consider antihistamines for itching"
        ]
    
    elif 'psoriasis' in disease_lower:
        return [
            "Keep skin moisturized",
            "Avoid triggers like stress and certain medications",
            "Consider medicated creams or phototherapy",
            "Consult dermatologist for treatment plan"
        ]
    
    elif 'fungal' in disease_lower or 'ringworm' in disease_lower:
        return [
            "Keep affected area clean and dry",
            "Use antifungal cream as directed",
            "Wash hands frequently to prevent spread",
            "Avoid sharing personal items"
        ]
    
    elif 'warts' in disease_lower or 'viral' in disease_lower:
        return [
            "Avoid touching or scratching",
            "Over-the-counter treatments available",
            "Consider cryotherapy from doctor",
            "Keep area covered to prevent spread"
        ]
    
    else:
        return [
            "Consult a qualified dermatologist for proper diagnosis",
            "Keep affected area clean and dry",
            "Avoid irritants and harsh chemicals",
            "Monitor for changes and seek help if worsens"
        ]

# ============================================================
# TEST WITH IMAGE
# ============================================================
print("\n" + "="*60)
print("📸 TEST PREDICTION")
print("="*60)

# Get test image path
test_image = input("\nEnter path to skin disease image\n(or press Enter to skip): ").strip()

# Remove quotes if user copied path with quotes
test_image = test_image.strip('"').strip("'")

if test_image and os.path.exists(test_image):
    try:
        print(f"\n🔄 Analyzing image...")
        print(f"   File: {os.path.basename(test_image)}")
        
        # Make prediction
        result = predict_skin_disease(test_image)
        
        if result['success']:
            results = result['results']
            original_size = result['original_size']
            
            print(f"   Original size: {original_size[0]}x{original_size[1]}")
            print(f"   Resized to: {img_width}x{img_height}")
            
            # Display results
            print("\n" + "="*60)
            print("📊 PREDICTION RESULTS")
            print("="*60)
            
            top_disease = results[0]['disease']
            top_confidence = results[0]['confidence']
            
            print(f"\n🔬 Top Prediction:")
            print(f"   Disease: {top_disease}")
            print(f"   Confidence: {top_confidence:.2f}%")
            
            # Confidence assessment
            if top_confidence < 30:
                print(f"   ⚠️  LOW confidence - Image may be unclear or outside training data")
            elif top_confidence < 50:
                print(f"   ⚠️  MEDIUM confidence - Review top 3 predictions below")
            else:
                print(f"   ✅ HIGH confidence prediction")
            
            # Risk level
            risk = get_risk_level(top_disease, top_confidence)
            print(f"\n⚠️  Risk Assessment:")
            print(f"   {risk}")
            
            # Top 5 predictions
            print(f"\n📋 Top 5 Predictions:")
            for i, res in enumerate(results, 1):
                disease = res['disease']
                confidence = res['confidence']
                
                # Truncate long names
                if len(disease) > 60:
                    disease = disease[:57] + "..."
                
                bar_length = int(confidence / 2)  # Scale to 50 chars max
                bar = "█" * bar_length
                
                print(f"\n   {i}. {disease}")
                print(f"      {bar} {confidence:.2f}%")
            
            # Recommendations
            recommendations = get_recommendations(top_disease)
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")
            
            # Disclaimer
            print("\n" + "="*60)
            print("⚠️  IMPORTANT DISCLAIMER")
            print("="*60)
            print("""
   This is an AI prediction for EDUCATIONAL PURPOSES ONLY.
   
   • NOT a substitute for professional medical advice
   • Always consult a qualified dermatologist
   • Do not make treatment decisions based solely on this
   • Accuracy may vary based on image quality
   • Model trained on limited dataset
            """)
            
        else:
            print(f"\n❌ Prediction failed: {result['error']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

elif test_image:
    print(f"\n❌ File not found: {test_image}")
    print("\n💡 Tips:")
    print("   • Make sure the path is correct")
    print("   • Use forward slashes: C:/Users/Name/image.jpg")
    print("   • Or backslashes: C:\\Users\\Name\\image.jpg")
    print("   • Path should not have quotes")
    print("\n📝 Example paths:")
    print("   C:\\Users\\Ankit Saraswat\\Downloads\\skin_image.jpg")
    print("   E:\\Deviathon\\tm2\\test_images\\sample.jpg")

else:
    print("\n✅ Model is ready for testing!")
    print("\n💡 To test with an image:")
    print("   1. Run: python test_skin_model.py")
    print("   2. Enter full path to your skin image")
    print("   3. Get instant predictions!")
    
    print("\n📱 Or use the web interface:")
    print("   1. Run: python app.py")
    print("   2. Open: http://localhost:5000")
    print("   3. Upload and analyze images in browser")
    
    print("\n📂 You can test with images from:")
    print("   • skin_disease_dataset/validation/")
    print("   • Any folder with skin images")

print("\n" + "="*60)
