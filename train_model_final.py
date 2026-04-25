# ====================================================================
# ENHANCED TRAIN MODEL - Uses Both CSV Files
# Symptom2Disease.csv + Symptom-severity.csv
# ====================================================================

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# LOAD SYMPTOM SEVERITY WEIGHTS
# ====================================================================

def load_symptom_weights(severity_csv='Symptom-severity.csv'):
    """Load symptom severity weights"""
    df_severity = pd.read_csv(severity_csv)
    # Create dictionary mapping symptom to weight
    severity_dict = dict(zip(df_severity['Symptom'], df_severity['weight']))
    print(f"✓ Loaded {len(severity_dict)} symptom severity weights")
    return severity_dict

# ====================================================================
# SYMPTOM LIST (131 symptoms)
# ====================================================================

SYMPTOMS_LIST = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
    'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
    'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
    'spotting_urination', 'fatigue', 'weight_gain', 'anxiety',
    'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
    'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
    'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
    'dehydration', 'indigestion', 'headache', 'yellowish_skin',
    'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
    'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
    'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure',
    'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes',
    'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
    'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
    'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps',
    'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
    'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
    'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain',
    'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
    'spinning_movements', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
    'foul_smell_of_urine', 'continuous_feel_of_urine', 'passage_of_gases',
    'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability',
    'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes',
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
    'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
    'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption',
    'blood_in_sputum', 'prominent_veins_on_calf',
    'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
    'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
    'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]

# ====================================================================
# TEXT TO SYMPTOM EXTRACTION
# ====================================================================

def extract_symptoms_from_text(text, severity_weights=None):
    """Extract symptoms from patient text description with severity weighting"""
    text_lower = text.lower()
    detected_symptoms = []
    
    # Comprehensive symptom keyword mappings
    symptom_keywords = {
        'itching': ['itch', 'itchy', 'scratching'],
        'skin_rash': ['rash', 'skin rash', 'red patches'],
        'nodal_skin_eruptions': ['eruption', 'nodule', 'skin bumps'],
        'continuous_sneezing': ['sneez', 'constant sneez'],
        'shivering': ['shiver', 'shivering', 'trembling'],
        'chills': ['chill', 'chills', 'cold shivers'],
        'joint_pain': ['joint pain', 'joints hurt', 'arthralgia', 'fingers', 'wrists', 'knees'],
        'stomach_pain': ['stomach pain', 'stomach ache', 'belly pain', 'abdominal discomfort'],
        'acidity': ['acidity', 'acid reflux', 'heartburn'],
        'vomiting': ['vomit', 'throw up', 'emesis'],
        'fatigue': ['tired', 'fatigue', 'exhausted', 'weakness'],
        'weight_loss': ['weight loss', 'losing weight', 'lost weight'],
        'weight_gain': ['weight gain', 'gaining weight'],
        'cough': ['cough', 'coughing'],
        'high_fever': ['high fever', 'high temperature'],
        'mild_fever': ['mild fever', 'slight fever', 'low grade fever'],
        'headache': ['headache', 'head pain'],
        'nausea': ['nausea', 'nauseated', 'feel sick'],
        'dizziness': ['dizzy', 'dizziness', 'lightheaded'],
        'breathlessness': ['breathless', 'short of breath', 'breathing difficulty'],
        'chest_pain': ['chest pain', 'pain in chest'],
        'back_pain': ['back pain', 'backache', 'lower back'],
        'neck_pain': ['neck pain', 'stiff neck', 'neck stiffness'],
        'constipation': ['constipat', 'difficult bowel'],
        'diarrhoea': ['diarr', 'loose stool', 'watery stool'],
        'skin_peeling': ['peeling', 'skin peeling', 'flaking skin', 'skin has been peeling'],
        'anxiety': ['anxious', 'anxiety', 'worried'],
        'depression': ['depress', 'sad', 'low mood'],
        'sweating': ['sweat', 'perspir'],
        'yellowing_of_eyes': ['yellow eyes', 'yellowing of eyes'],
        'yellowish_skin': ['yellow skin', 'yellowish skin', 'jaundice'],
        'muscle_pain': ['muscle pain', 'myalgia', 'muscle ache'],
        'loss_of_appetite': ['loss of appetite', 'no appetite', 'not hungry'],
        'dehydration': ['dehydrat', 'dry'],
        'indigestion': ['indigestion', 'dyspepsia'],
        'blurred_and_distorted_vision': ['blurred vision', 'vision problem', 'visual'],
        'watering_from_eyes': ['watery eyes', 'tearing'],
        'red_spots_over_body': ['red spots', 'spots on body'],
        'pus_filled_pimples': ['pimples', 'acne', 'pus'],
        'blackheads': ['blackhead'],
        'scurring': ['scar', 'scarring'],
        'silver_like_dusting': ['dusting', 'silver dust', 'scales', 'silver like dusting'],
        'small_dents_in_nails': ['dents in nails', 'nail pits', 'pitted nails', 'small dents', 'dents'],
        'inflammatory_nails': ['nail inflammation', 'inflammatory nails', 'inflamed nails', 'inflammatory', 'tender'],
        'blister': ['blister'],
        'red_sore_around_nose': ['sore around nose', 'red nose'],
        'yellow_crust_ooze': ['crust', 'ooze', 'discharge'],
        'cramps': ['cramp'],
        'bruising': ['bruis'],
        'swollen_legs': ['swollen leg', 'leg swelling'],
        'swelling_joints': ['swollen joint', 'joint swelling'],
        'knee_pain': ['knee pain', 'knees'],
        'hip_joint_pain': ['hip pain'],
        'painful_walking': ['painful walking', 'pain when walking'],
        'movement_stiffness': ['stiff', 'stiffness'],
        'loss_of_balance': ['loss of balance', 'unbalanced'],
        'spinning_movements': ['spinning', 'vertigo'],
        'unsteadiness': ['unsteady'],
        'palpitations': ['palpitation', 'heart racing'],
        'fast_heart_rate': ['fast heart', 'rapid heart'],
        'abdominal_pain': ['abdominal pain'],
        'belly_pain': ['belly pain'],
        'patches_in_throat': ['throat patches', 'patches in throat'],
        'throat_irritation': ['throat irritation', 'sore throat'],
        'runny_nose': ['runny nose', 'nasal discharge'],
        'congestion': ['congest', 'blocked nose'],
        'sinus_pressure': ['sinus'],
        'phlegm': ['phlegm', 'mucus'],
        'redness_of_eyes': ['red eyes'],
        'dark_urine': ['dark urine'],
        'yellow_urine': ['yellow urine'],
        'burning_micturition': ['burning urination', 'pain while urinating'],
        'spotting_urination': ['blood in urine', 'spotting urine'],
        'foul_smell_of_urine': ['foul smell urine'],
        'continuous_feel_of_urine': ['frequent urination', 'urge to urinate'],
        'passage_of_gases': ['gas', 'flatulence'],
        'internal_itching': ['internal itch'],
        'mood_swings': ['mood swing'],
        'restlessness': ['restless'],
        'lethargy': ['lethargic', 'lethargy'],
        'irregular_sugar_level': ['sugar level', 'blood sugar'],
        'obesity': ['obese', 'obesity', 'overweight'],
        'excessive_hunger': ['excessive hunger', 'always hungry'],
        'increased_appetite': ['increased appetite'],
        'polyuria': ['polyuria', 'excessive urination'],
        'visual_disturbances': ['visual disturbance'],
        'altered_sensorium': ['confusion', 'altered consciousness'],
        'family_history': ['family history'],
        'mucoid_sputum': ['mucoid sputum'],
        'rusty_sputum': ['rusty sputum'],
        'lack_of_concentration': ['can\'t concentrate', 'poor concentration'],
        'receiving_blood_transfusion': ['blood transfusion'],
        'receiving_unsterile_injections': ['injection'],
        'coma': ['coma', 'unconscious'],
        'stomach_bleeding': ['stomach bleeding'],
        'distention_of_abdomen': ['abdominal distention', 'bloating'],
        'history_of_alcohol_consumption': ['alcohol'],
        'blood_in_sputum': ['blood in sputum', 'coughing blood'],
        'prominent_veins_on_calf': ['prominent veins', 'varicose'],
        'malaise': ['malaise', 'unwell'],
        'pain_behind_the_eyes': ['pain behind eyes'],
        'sunken_eyes': ['sunken eyes'],
        'acute_liver_failure': ['liver failure'],
        'fluid_overload': ['fluid overload', 'edema'],
        'swelling_of_stomach': ['stomach swelling'],
        'swelled_lymph_nodes': ['swollen lymph', 'lymph nodes'],
        'pain_during_bowel_movements': ['pain during bowel'],
        'pain_in_anal_region': ['anal pain'],
        'bloody_stool': ['blood in stool', 'bloody stool'],
        'irritation_in_anus': ['anal irritation'],
        'cold_hands_and_feets': ['cold hands', 'cold feet'],
        'puffy_face_and_eyes': ['puffy face', 'swollen face'],
        'enlarged_thyroid': ['enlarged thyroid', 'goiter'],
        'brittle_nails': ['brittle nails'],
        'swollen_extremeties': ['swollen extremities'],
        'extra_marital_contacts': ['extra marital'],
        'drying_and_tingling_lips': ['dry lips', 'tingling lips'],
        'slurred_speech': ['slurred speech'],
        'muscle_weakness': ['muscle weakness'],
        'stiff_neck': ['stiff neck'],
        'weakness_of_one_body_side': ['one side weakness', 'hemiparesis'],
        'loss_of_smell': ['loss of smell', 'anosmia'],
        'bladder_discomfort': ['bladder discomfort'],
        'toxic_look_(typhos)': ['toxic look'],
        'weakness_in_limbs': ['limb weakness'],
        'abnormal_menstruation': ['irregular period', 'abnormal menstruation'],
        'dischromic_patches': ['patches on skin', 'discoloration'],
        'ulcers_on_tongue': ['tongue ulcer', 'ulcers on tongue'],
        'muscle_wasting': ['muscle wasting'],
        'swollen_blood_vessels': ['swollen vessels'],
    }
    
    # Check each symptom
    for symptom, keywords in symptom_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected_symptoms.append(symptom)
                break
    
    return detected_symptoms

def preprocess_text_to_binary(df, severity_weights):
    """Convert text descriptions to binary symptom matrix with optional severity weighting"""
    print("Converting text descriptions to binary symptom features...")
    
    # Initialize symptom matrix
    symptom_matrix = pd.DataFrame(0, index=df.index, columns=SYMPTOMS_LIST)
    
    # Process each row
    for idx, row in df.iterrows():
        text = row['text']
        detected = extract_symptoms_from_text(text, severity_weights)
        
        # Set detected symptoms to 1 (or use severity weight if available)
        for symptom in detected:
            if symptom in SYMPTOMS_LIST:
                # Use binary (1) or severity weight
                weight = severity_weights.get(symptom, 1) if severity_weights else 1
                symptom_matrix.at[idx, symptom] = 1  # Keep binary for now
    
    # Add disease label
    symptom_matrix['prognosis'] = df['label']
    
    avg_symptoms = symptom_matrix[SYMPTOMS_LIST].sum(axis=1).mean()
    print(f"✓ Processed {len(df)} records")
    print(f"✓ Average symptoms per record: {avg_symptoms:.2f}")
    
    return symptom_matrix

# ====================================================================
# MAIN TRAINING FUNCTION
# ====================================================================

def train_model_with_both_files(
    disease_csv='Symptom2Disease.csv',
    severity_csv='Symptom-severity.csv'
):
    """Train model using both CSV files"""
    
    print("="*80)
    print("DISEASE PREDICTION MODEL TRAINING")
    print("Using Symptom2Disease.csv + Symptom-severity.csv")
    print("="*80)
    
    # Load symptom severity weights
    print(f"\n1. Loading symptom severity weights from {severity_csv}...")
    severity_weights = load_symptom_weights(severity_csv)
    
    # Load disease-symptom data
    print(f"\n2. Loading disease data from {disease_csv}...")
    df = pd.read_csv(disease_csv)
    print(f"✓ Loaded {len(df)} records with {df['label'].nunique()} diseases")
    print(f"✓ Diseases: {sorted(df['label'].unique().tolist())}")
    
    # Convert text to binary
    print("\n3. Converting text descriptions to binary features...")
    binary_data = preprocess_text_to_binary(df, severity_weights)
    
    # Prepare features and labels
    print("\n4. Preparing training data...")
    X = binary_data[SYMPTOMS_LIST]
    y = binary_data['prognosis']
    
    # Encode labels
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Number of diseases: {len(le_target.classes_)}")
    
    # Split data
    print("\n5. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Testing samples: {len(X_test)}")
    
    # Scale features
    print("\n6. Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features scaled")
    
    # Train model
    print("\n7. Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False,
        verbosity=0
    )
    
    model.fit(X_train_scaled, y_train)
    print("✓ Model training complete")
    
    # Evaluate
    print("\n8. Evaluating model performance...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\n{'='*80}")
    print(f"  TRAINING ACCURACY: {train_accuracy*100:.2f}%")
    print(f"  TESTING ACCURACY:  {test_accuracy*100:.2f}%")
    print(f"{'='*80}\n")
    
    # Create explainability models
    print("9. Creating SHAP explainer...")
    shap_explainer = shap.TreeExplainer(model)
    print("✓ SHAP explainer created")
    
    # Save everything
    print("\n10. Saving model and artifacts...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save main model package
    model_package = {
        'model': model,
        'scaler': scaler,
        'le_target': le_target,
        'shap_explainer': shap_explainer,
        'feature_names': SYMPTOMS_LIST,
        'severity_weights': severity_weights,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }
    
    with open('models/disease_prediction_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    
    print("✓ Model saved to models/disease_prediction_model.pkl")
    
    # Save individual components (for compatibility)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le_target, f)
    
    with open('models/shap_explainer.pkl', 'wb') as f:
        pickle.dump(shap_explainer, f)
    
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(SYMPTOMS_LIST, f)
    
    print("✓ Individual components also saved")
    
    print(f"\n{'='*80}")
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print("\nNext steps:")
    print("1. Run your Flask app: python app.py")
    print("2. Navigate to http://localhost:5000")
    print("3. Test predictions with symptom selection")
    
    return model, scaler, le_target, test_accuracy

# ====================================================================
# MAIN EXECUTION
# ====================================================================

if __name__ == "__main__":
    print("\nStarting model training with both CSV files...\n")
    
    # Train the model
    model, scaler, le_target, accuracy = train_model_with_both_files(
        disease_csv='Symptom2Disease.csv',
        severity_csv='Symptom-severity.csv'
    )
    
    print(f"\n✓ Model is ready to use!")
    print(f"✓ Final Testing Accuracy: {accuracy*100:.2f}%\n")
