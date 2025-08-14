# ===================================================================
# SCRIPT: debug_intent_mapping.py (FINAL - Support CSV)
# ===================================================================
import torch
import json
import pandas as pd
from pathlib import Path

def debug_intent_model():
    """
    Debug intent model dan extract mapping yang benar dari model checkpoint
    """
    print("🔍 Debugging Intent Classification Model\n")

    model_path = Path("models/intent_classifier/best_model.pth")

    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return

    try:
        checkpoint = torch.load(model_path, map_location='cpu')

        print("📦 Checkpoint contents:")
        for key in checkpoint.keys():
            print(f"   - {key}: {type(checkpoint[key])}")

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"\n🏗️ Model state dict keys:")
            for key in state_dict.keys():
                if 'classifier' in key:
                    print(f"   - {key}: {state_dict[key].shape}")

        for key, value in state_dict.items():
            if 'classifier.weight' in key:
                num_classes = value.shape[0]
                print(f"\n🎯 Number of classes from model architecture: {num_classes}")
                break

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")

def check_training_data_csv():
    """
    Ekstrak mapping dari training data berbentuk CSV
    """
    print("\n📚 Checking training data structure (CSV mode)...")

    training_data_path = Path("data/training_data/intent_data.csv")

    if training_data_path.exists():
        try:
            df = pd.read_csv(training_data_path)
            print(f"📊 Loaded {len(df)} rows from training data")

            if "intent" not in df.columns:
                print("❌ Kolom 'intent' tidak ditemukan di CSV!")
                return

            # Ambil unique intents
            unique_intents = sorted(df["intent"].unique().tolist())
            print(f"🎯 Unique intents found ({len(unique_intents)}):")
            for i, intent in enumerate(unique_intents):
                print(f"   {i}: {intent}")

            # Buat mapping id → intent
            intent_mapping = {i: intent for i, intent in enumerate(unique_intents)}
            mapping_path = Path("models/intent_classifier/intent_mapping_extracted.json")
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump({"id_to_intent": {str(k): v for k, v in intent_mapping.items()}},
                          f, indent=2, ensure_ascii=False)

            print(f"\n💾 Intent mapping extracted & saved to: {mapping_path}")

        except Exception as e:
            print(f"❌ Error reading CSV training data: {e}")
    else:
        print(f"❌ Training data not found at {training_data_path}")

def extract_from_notebook():
    """
    Petunjuk manual jika perlu buka notebook
    """
    print("\n📓 MANUAL EXTRACTION from Jupyter Notebook:")
    print("1. Open 'Intent Classifier Training Pipeline.ipynb'")
    print("2. Look for output seperti 'Label Encoder Classes' atau 'Intent Mapping'")
    print("3. Copy mapping {0: 'BATAS_SKS', 1: 'DOSEN_PENGAMPU', ...}")

if __name__ == "__main__":
    debug_intent_model()
    check_training_data_csv()
    extract_from_notebook()
