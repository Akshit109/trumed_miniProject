import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import os

# ============================================================
# CONFIGURATION - FIXED TO 160x160
# ============================================================
MODEL_PATH = 'models/skin_disease_model.h5'
CLASSES_PATH = 'models/skin_class_names.json'
IMG_SIZE = (160, 160)  # ✅ FIXED - Must match training size!

class SkinDiseasePredictor:
    def __init__(self):
        self.model = None
        self.disease_classes = {}
        self.load_model()
    
    def load_model(self):
        """Load the trained model and class names"""
        try:
            if os.path.exists(MODEL_PATH):
                print(f"Loading model from {MODEL_PATH}...")
                self.model = load_model(MODEL_PATH)
                print(f"✅ Model loaded! Input shape: {self.model.input_shape}")
                
                if os.path.exists(CLASSES_PATH):
                    with open(CLASSES_PATH, 'r') as f:
                        self.disease_classes = json.load(f)
                    print(f"✅ Loaded {len(self.disease_classes)} disease classes")
                else:
                    print(f"⚠️ Warning: {CLASSES_PATH} not found")
                    # Create dummy classes if file missing
                    self.disease_classes = {str(i): f"Disease_{i}" for i in range(23)}
            else:
                print(f"❌ Model not found at {MODEL_PATH}")
                print("   Using demo mode")
                self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to exact model input size
            img_resized = image.resize(IMG_SIZE)
            
            # Convert to array and normalize to [0, 1]
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            print(f"✅ Image preprocessed: {img_array.shape}")
            return img_array
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            raise
    
    def predict(self, image):
        """Predict skin disease from image"""
        
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not loaded. Please train the model first.',
                'disease': 'Demo Disease',
                'confidence': 50.0,
                'risk_level': 'Low',
                'recommendations': ['This is demo mode', 'Train model to enable predictions'],
                'demo_mode': True
            }
        
        try:
            # Preprocess image
            img_array = self.preprocess_image(image)
            
            # Make prediction
            print("Making prediction...")
            predictions = self.model.predict(img_array, verbose=0)
            print(f"✅ Prediction complete! Shape: {predictions.shape}")
            
            # Get top 5 predictions
            top5_indices = np.argsort(predictions[0])[-5:][::-1]
            
            # Get top prediction
            top_idx = top5_indices[0]
            top_disease = self.disease_classes.get(str(top_idx), f"Disease_{top_idx}")
            top_confidence = float(predictions[0][top_idx] * 100)
            
            # Get all top 5 predictions
            all_predictions = {}
            for idx in top5_indices:
                disease = self.disease_classes.get(str(idx), f"Disease_{idx}")
                confidence = float(predictions[0][idx] * 100)
                all_predictions[disease] = confidence
            
            # Determine risk level
            risk_level = self.get_risk_level(top_disease, top_confidence)
            
            # Get recommendations
            recommendations = self.get_recommendations(top_disease)
            
            print(f"✅ Top prediction: {top_disease} ({top_confidence:.2f}%)")
            
            return {
                'success': True,
                'disease': top_disease,
                'confidence': round(top_confidence, 2),
                'risk_level': risk_level,
                'recommendations': recommendations,
                'all_predictions': all_predictions
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f"Prediction failed: {str(e)}"
            }
    
    def get_risk_level(self, disease_name, confidence):
        """Determine risk level based on disease"""
        
        high_risk_keywords = [
            'cancer', 'carcinoma', 'melanoma', 'malignant', 
            'lesions', 'basal cell'
        ]
        
        medium_risk_keywords = [
            'infection', 'bacterial', 'fungal', 'viral',
            'cellulitis', 'impetigo', 'herpes'
        ]
        
        disease_lower = disease_name.lower()
        
        # Check for high risk conditions
        for keyword in high_risk_keywords:
            if keyword in disease_lower:
                return 'High'
        
        # Check for medium risk conditions
        for keyword in medium_risk_keywords:
            if keyword in disease_lower:
                return 'Medium'
        
        # Low confidence = uncertain
        if confidence < 40:
            return 'Uncertain'
        
        return 'Low'
    
    def get_recommendations(self, disease_name):
        """Get recommendations based on disease"""
        
        disease_lower = disease_name.lower()
        
        if 'acne' in disease_lower or 'rosacea' in disease_lower:
            return [
                "Keep skin clean with gentle, non-comedogenic cleanser",
                "Avoid touching or picking affected areas",
                "Consider over-the-counter benzoyl peroxide or salicylic acid",
                "Consult dermatologist for persistent or severe cases",
                "Maintain consistent skincare routine"
            ]
        
        elif 'cancer' in disease_lower or 'melanoma' in disease_lower or 'carcinoma' in disease_lower:
            return [
                "⚠️ URGENT: Schedule appointment with dermatologist IMMEDIATELY",
                "Avoid sun exposure and use broad-spectrum SPF 50+ sunscreen",
                "Do not attempt self-treatment",
                "Get professional biopsy and pathology examination",
                "Early detection is critical for successful treatment"
            ]
        
        elif 'eczema' in disease_lower or 'dermatitis' in disease_lower:
            return [
                "Moisturize regularly with fragrance-free, hypoallergenic products",
                "Identify and avoid known triggers (stress, allergens, irritants)",
                "Use gentle, soap-free cleansers",
                "Consider over-the-counter hydrocortisone cream for mild cases",
                "Consult dermatologist if symptoms persist or worsen"
            ]
        
        elif 'psoriasis' in disease_lower:
            return [
                "Keep skin well-moisturized with thick creams or ointments",
                "Avoid triggers like stress, alcohol, and smoking",
                "Consider coal tar or salicylic acid topical treatments",
                "Consult dermatologist for prescription medications or phototherapy",
                "Join support groups for coping strategies"
            ]
        
        elif 'fungal' in disease_lower or 'ringworm' in disease_lower or 'tinea' in disease_lower:
            return [
                "Keep affected area clean and completely dry",
                "Apply over-the-counter antifungal cream (clotrimazole, terbinafine)",
                "Wash hands frequently to prevent spreading",
                "Avoid sharing towels, clothing, or personal items",
                "See doctor if not improved after 2 weeks of treatment"
            ]
        
        elif 'warts' in disease_lower or 'viral' in disease_lower or 'molluscum' in disease_lower:
            return [
                "Avoid touching, scratching, or picking at lesions",
                "Over-the-counter salicylic acid treatments available",
                "Consider cryotherapy (freezing) from healthcare provider",
                "Keep area covered to prevent spreading to others",
                "May resolve on their own but can take months"
            ]
        
        elif 'hives' in disease_lower or 'urticaria' in disease_lower:
            return [
                "Identify and avoid triggers (foods, medications, stress)",
                "Take over-the-counter antihistamines (cetirizine, loratadine)",
                "Apply cool compresses to affected areas for relief",
                "Wear loose, breathable clothing",
                "Seek emergency care if breathing difficulty or throat swelling"
            ]
        
        else:
            return [
                "Consult a qualified dermatologist for accurate diagnosis and treatment",
                "Keep affected area clean and dry",
                "Avoid harsh chemicals, fragrances, and irritants",
                "Monitor for any changes in size, color, or symptoms",
                "Take clear photos to track progression over time"
            ]

# Create global predictor instance
print("="*60)
print("Initializing Skin Disease Predictor...")
print("="*60)
skin_predictor = SkinDiseasePredictor()
print("="*60)
