import os
import shutil

print("="*80)
print("ORGANIZING SKIN DISEASE DATASET")
print("="*80)

source = "raw_dataset/IMG_CLASSES"
target = "skin_dataset"

os.makedirs(target, exist_ok=True)

# Correct folder mapping based on actual folder names
folder_map = {
    '1. Eczema 1677': 'Eczema',
    '3. Atopic Dermatitis - 1.25k': 'Eczema',  # Merge with Eczema
    '2. Melanoma 15.75k': 'Melanoma',
    '4. Basal Cell Carcinoma (BCC) 3323': 'Basal Cell Carcinoma',
    '7. Psoriasis pictures Lichen Planus and related diseases - 2k': 'Psoriasis',
    '10. Warts Molluscum and other Viral Infections - 2103': 'Warts',
    '9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k': 'Fungal Infection',
    '5. Melanocytic Nevi (NV) - 7970': 'Normal Skin',  # Benign moles = normal
    '6. Benign Keratosis-like Lesions (BKL) 2624': 'Dermatofibroma',
    '8. Seborrheic Keratoses and other Benign Tumors - 1.8k': 'Dermatofibroma',
}

print("\nCopying images to organized folders...")

for old_name, new_name in folder_map.items():
    old_path = os.path.join(source, old_name)
    new_path = os.path.join(target, new_name)
    
    if os.path.exists(old_path):
        os.makedirs(new_path, exist_ok=True)
        files = os.listdir(old_path)
        count = 0
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                try:
                    src = os.path.join(old_path, file)
                    dst = os.path.join(new_path, file)
                    shutil.copy2(src, dst)
                    count += 1
                except Exception as e:
                    pass
        print(f"✓ {new_name}: {count} images")

# Create missing folders
missing = ['Acne', 'Vitiligo', 'Rosacea']
print("\nCreating placeholder folders:")
for cls in missing:
    path = os.path.join(target, cls)
    os.makedirs(path, exist_ok=True)
    print(f"⚠ {cls}: 0 images (not in dataset)")

print("\n" + "="*80)
print("✓ DATASET ORGANIZATION COMPLETE!")
print("="*80)

print("\n📊 Final Dataset Statistics:")
total = 0
for folder in sorted(os.listdir(target)):
    folder_path = os.path.join(target, folder)
    if os.path.isdir(folder_path):
        count = len([f for f in os.listdir(folder_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        total += count
        print(f"  {folder}: {count} images")

print(f"\n✓ Total images: {total}")
print("\nReady for training! Run: python train_skin_model.py")