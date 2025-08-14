# ===================================================================
# FILE: fix_intent_mapping.py
# ===================================================================
# Script untuk create intent mapping yang benar berdasarkan training output

import json
from pathlib import Path

def create_correct_intent_mapping():
    """
    Create intent mapping yang benar berdasarkan training data analysis
    """
    print("🔧 Creating Correct Intent Mapping...")
    
    # Berdasarkan analysis dari training output dan debug data
    # Setelah removing classes 51 & 53, mapping yang benar adalah:
    correct_mapping = {
        "0": "BATAS_SKS",
        "1": "CLARIFICATION", 
        "2": "DOSEN_PENGAMPU",
        "3": "GOODBYE",
        "4": "GREETING",
        "5": "HELP",
        "6": "INFO_DOSEN_UMUM",
        "7": "INFO_MATAKULIAH",
        "8": "JADWAL_DOSEN",
        "9": "JADWAL_HARI",
        "10": "JADWAL_KRS",
        "11": "JADWAL_MATAKULIAH",
        "12": "JADWAL_PRODI",
        "13": "JADWAL_RUANGAN",
        "14": "JADWAL_SEMESTER",
        "15": "KONTAK_DOSEN",
        "16": "NIDN_DOSEN",
        "17": "OUT_OF_SCOPE",
        "18": "PANDUAN_KRS",
        "19": "PRASYARAT_MATKUL",
        "20": "PROSEDUR_CUTI",
        "21": "SKS_MATKUL",
        "22": "SYARAT_SKRIPSI"
    }
    
    # Verify total classes match model architecture
    assert len(correct_mapping) == 23, f"Mapping should have 23 classes, got {len(correct_mapping)}"
    
    # Save correct mapping
    mapping_path = Path("models/intent_classifier/intent_mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(correct_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Correct intent mapping saved to: {mapping_path}")
    print(f"📊 Total classes: {len(correct_mapping)}")
    
    # Print mapping untuk verification
    print("\n📋 Intent Mapping:")
    for class_id, intent_name in correct_mapping.items():
        print(f"   {class_id}: {intent_name}")
    
    return correct_mapping

def backup_old_mapping():
    """
    Backup old mapping sebelum replace
    """
    old_path = Path("models/intent_classifier/intent_mapping.json")
    backup_path = Path("models/intent_classifier/intent_mapping_old.json")
    
    if old_path.exists():
        import shutil
        shutil.copy2(old_path, backup_path)
        print(f"💾 Old mapping backed up to: {backup_path}")

def verify_mapping_with_model():
    """
    Verify mapping compatibility dengan model architecture
    """
    import torch
    
    model_path = Path("models/intent_classifier/best_model.pth")
    if not model_path.exists():
        print("❌ Model file not found for verification")
        return False
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        # Get classifier layer size
        classifier_weight = state_dict['classifier.weight']
        num_classes_in_model = classifier_weight.shape[0]
        
        print(f"🏗️ Model expects {num_classes_in_model} classes")
        
        # Load our mapping
        mapping_path = Path("models/intent_classifier/intent_mapping.json")
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        num_classes_in_mapping = len(mapping)
        print(f"📋 Mapping provides {num_classes_in_mapping} classes")
        
        if num_classes_in_model == num_classes_in_mapping:
            print("✅ Mapping matches model architecture!")
            return True
        else:
            print("❌ Mapping does NOT match model architecture!")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying mapping: {e}")
        return False

if __name__ == "__main__":
    print("🔧 FIXING INTENT MAPPING\n")
    
    # Step 1: Backup old mapping
    backup_old_mapping()
    
    # Step 2: Create correct mapping
    correct_mapping = create_correct_intent_mapping()
    
    # Step 3: Verify compatibility
    is_compatible = verify_mapping_with_model()
    
    if is_compatible:
        print("\n🎉 SUCCESS: Intent mapping fixed!")
        print("🚀 Restart your application to apply changes")
    else:
        print("\n❌ ISSUE: Mapping verification failed")
        print("🔍 Check model file and mapping manually")

# ===================================================================
# ADDITIONAL FIX: Update Intent Service untuk handle reload
# ===================================================================

def test_intent_predictions():
    """
    Test predictions dengan mapping yang baru
    """
    print("\n🧪 Testing Predictions with New Mapping...")
    
    # Import and test (jika service sudah running)
    try:
        import sys
        sys.path.append('.')
        
        from app.services.intent_service import IntentClassificationService
        
        # Create new instance dengan mapping yang baru
        service = IntentClassificationService()
        success = service.load_model()
        
        if success:
            # Test samples
            test_queries = [
                "Siapa dosen machine learning?",
                "Jadwal kuliah hari senin gimana?", 
                "Syarat skripsi apa saja?",
                "Berapa batas SKS semester 3?",
                "Kontak pak yuda berapa?"
            ]
            
            print("🎯 Test Results:")
            for query in test_queries:
                intent, confidence = service.predict_intent(query)
                print(f"   '{query}' → {intent} ({confidence:.4f})")
                
        else:
            print("❌ Failed to load service with new mapping")
            
    except Exception as e:
        print(f"⚠️ Cannot test automatically: {e}")
        print("💡 Restart application manually to test")

# ===================================================================
# FINAL VERIFICATION BASED ON TRAINING OUTPUT
# ===================================================================

def create_mapping_from_training_output():
    """
    Create mapping berdasarkan exact training output yang diberikan
    """
    print("\n🎯 Creating mapping based on EXACT training output...")
    
    # Dari training output, intent yang valid (setelah remove 51, 53):
    # Urutan alphabetical seperti yang ditunjukkan di classification report
    training_based_mapping = {
        "0": "BATAS_SKS",
        "1": "CLARIFICATION", 
        "2": "DOSEN_PENGAMPU",
        "3": "GOODBYE",
        "4": "GREETING",
        "5": "HELP",
        "6": "INFO_DOSEN_UMUM",
        "7": "INFO_MATAKULIAH",
        "8": "JADWAL_DOSEN",
        "9": "JADWAL_HARI",
        "10": "JADWAL_KRS",
        "11": "JADWAL_MATAKULIAH",
        "12": "JADWAL_PRODI",
        "13": "JADWAL_RUANGAN",
        "14": "JADWAL_SEMESTER",
        "15": "KONTAK_DOSEN",
        "16": "NIDN_DOSEN",
        "17": "OUT_OF_SCOPE",
        "18": "PANDUAN_KRS",
        "19": "PRASYARAT_MATKUL",
        "20": "PROSEDUR_CUTI",
        "21": "SKS_MATKUL",
        "22": "SYARAT_SKRIPSI"
    }
    
    # Save sebagai final mapping
    final_path = Path("models/intent_classifier/intent_mapping_final.json")
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(training_based_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Final mapping saved to: {final_path}")
    
    # Copy sebagai main mapping
    main_path = Path("models/intent_classifier/intent_mapping.json")
    with open(main_path, 'w', encoding='utf-8') as f:
        json.dump(training_based_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Main mapping updated: {main_path}")
    
    return training_based_mapping