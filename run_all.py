#!/usr/bin/env python3
"""
Master script to run MELAUDIS processing steps.
Your data already has audio files, so this skips extraction and goes to spectrogram generation.
"""

import os
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories with your split audio data
TRAIN_AUDIO_PATH = "data/train/"
TEST_AUDIO_PATH = "data/test/"
SPECTROGRAM_OUTPUT_PATH = "data/spectrograms/"

# Spectrogram paths (separated by train/test)
TRAIN_SPEC_PATH = "data/spectrograms/train/"
TEST_SPEC_PATH = "data/spectrograms/test/"

# Audio extraction parameters
EXTRACTION_RANGE_SEC = 1.0  # Used if you add extraction steps later

# Limit files for testing (set to None for full dataset)
MAX_FILES_TO_PROCESS = 100  # For testing: process only 100 BG + 100 Veh per set

# ============================================================================
# EXECUTION STEPS
# ============================================================================

def step_1_extract_vehicle_audio():
    """SKIPPED - You already have extracted audio files"""
    print("✓ Skipping vehicle extraction (audio already extracted)")
    return True

def step_2_extract_background_audio():
    """SKIPPED - You already have background audio files"""
    print("✓ Skipping background extraction (audio already extracted)")
    return True

def step_3_generate_spectrograms():
    """Generate mel-spectrograms from training and testing audio files"""
    print("\n" + "="*60)
    print("STEP 3: Generating Spectrograms (Train & Test)")
    print("="*60)
    
    try:
        import librosa
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        # Create output structure
        for spec_path in [TRAIN_SPEC_PATH, TEST_SPEC_PATH]:
            os.makedirs(f"{spec_path}BG", exist_ok=True)
            os.makedirs(f"{spec_path}Veh", exist_ok=True)
        
        FRAME_SIZE = 512
        HOP_SIZE = 64
        
        def save_spec(audio, sr, save_path):
            """Generate and save spectrogram"""
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            
            plt.figure(figsize=(5, 3))
            librosa.display.specshow(log_mel, sr=sr, hop_length=HOP_SIZE, y_axis='mel')
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        def process_audio_set(audio_base_path, spec_base_path, set_name):
            """Process all audio files in a set (train or test)"""
            print(f"\n  {set_name} Set:")
            
            # Process background
            print(f"    Processing background...")
            bg_count = 0
            bg_path = f"{audio_base_path}bg_noise/"
            if os.path.exists(bg_path):
                for file in os.listdir(bg_path):
                    if MAX_FILES_TO_PROCESS and bg_count >= MAX_FILES_TO_PROCESS:
                        break
                    if file.endswith('.wav'):
                        try:
                            audio, sr = librosa.load(os.path.join(bg_path, file), sr=None)
                            save_path = f"{spec_base_path}BG/{file[:-4]}.png"
                            save_spec(audio, sr, save_path)
                            bg_count += 1
                        except Exception as e:
                            print(f"      Error processing {file}: {e}")
            
            # Process vehicles
            print(f"    Processing vehicles...")
            veh_count = 0
            veh_path = f"{audio_base_path}vehicles/"
            if os.path.exists(veh_path):
                for file in os.listdir(veh_path):
                    if MAX_FILES_TO_PROCESS and veh_count >= MAX_FILES_TO_PROCESS:
                        break
                    if file.endswith('.wav'):
                        try:
                            audio, sr = librosa.load(os.path.join(veh_path, file), sr=None)
                            save_path = f"{spec_base_path}Veh/{file[:-4]}.png"
                            save_spec(audio, sr, save_path)
                            veh_count += 1
                        except Exception as e:
                            print(f"      Error processing {file}: {e}")
            
            print(f"    ✓ {set_name} spectrograms: {bg_count} BG + {veh_count} Veh")
            return bg_count, veh_count
        
        # Process both train and test
        train_bg, train_veh = process_audio_set(TRAIN_AUDIO_PATH, TRAIN_SPEC_PATH, "Train")
        test_bg, test_veh = process_audio_set(TEST_AUDIO_PATH, TEST_SPEC_PATH, "Test")
        
        print(f"\n✓ Total spectrograms generated:")
        print(f"  Train: {train_bg + train_veh} ({train_bg} BG + {train_veh} Veh)")
        print(f"  Test: {test_bg + test_veh} ({test_bg} BG + {test_veh} Veh)")
        return True
    except Exception as e:
        print(f"✗ Error in spectrogram generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def step_4_train_classifier():
    """Train on training data, evaluate on test data"""
    print("\n" + "="*60)
    print("STEP 4: Training & Evaluating Model")
    print("="*60)
    
    try:
        import cv2
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Check if spectrogram directories exist
        if not os.path.exists(TRAIN_SPEC_PATH) or not os.path.exists(TEST_SPEC_PATH):
            print("⚠ Spectrogram directories not found. Run step 3 first.")
            return False
        
        # Load TRAINING data
        print("\n📚 Loading TRAINING data...")
        train_images = []
        train_labels = []
        
        for file in os.listdir(f"{TRAIN_SPEC_PATH}BG"):
            if file.endswith('.png'):
                img = cv2.imread(os.path.join(f"{TRAIN_SPEC_PATH}BG", file))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (223, 163))
                    train_images.append(img)
                    train_labels.append(0)  # BG = 0
        
        for file in os.listdir(f"{TRAIN_SPEC_PATH}Veh"):
            if file.endswith('.png'):
                img = cv2.imread(os.path.join(f"{TRAIN_SPEC_PATH}Veh", file))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (223, 163))
                    train_images.append(img)
                    train_labels.append(1)  # Veh = 1
        
        if len(train_images) == 0:
            print("⚠ No training spectrogram images found.")
            return False
        
        train_images = np.array(train_images, dtype='float32') / 255.0
        train_labels = np.array(train_labels, dtype='int32')
        print(f"  ✓ Loaded {len(train_images)} training images")
        print(f"    - Background: {sum(train_labels == 0)}")
        print(f"    - Vehicle: {sum(train_labels == 1)}")
        
        # Load TEST data
        print("\n🧪 Loading TEST data...")
        test_images = []
        test_labels = []
        
        for file in os.listdir(f"{TEST_SPEC_PATH}BG"):
            if file.endswith('.png'):
                img = cv2.imread(os.path.join(f"{TEST_SPEC_PATH}BG", file))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (223, 163))
                    test_images.append(img)
                    test_labels.append(0)
        
        for file in os.listdir(f"{TEST_SPEC_PATH}Veh"):
            if file.endswith('.png'):
                img = cv2.imread(os.path.join(f"{TEST_SPEC_PATH}Veh", file))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (223, 163))
                    test_images.append(img)
                    test_labels.append(1)
        
        if len(test_images) == 0:
            print("⚠ No test spectrogram images found.")
            return False
        
        test_images = np.array(test_images, dtype='float32') / 255.0
        test_labels = np.array(test_labels, dtype='int32')
        print(f"  ✓ Loaded {len(test_images)} test images")
        print(f"    - Background: {sum(test_labels == 0)}")
        print(f"    - Vehicle: {sum(test_labels == 1)}")
        
        # Build and train model
        print("\n🤖 Building model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(163, 223, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        print("📖 Training on training data...")
        history = model.fit(
            train_images, train_labels,
            epochs=5, batch_size=16,
            validation_split=0.2, verbose=1
        )
        
        # Evaluate on test data
        print("\n📊 Evaluating on TEST data...")
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
        print(f"✓ Test Accuracy: {test_accuracy*100:.1f}%")
        
        # Detailed evaluation
        print("\n📈 Detailed Metrics:")
        test_pred = (model.predict(test_images) > 0.5).astype(int).flatten()
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(test_labels, test_pred)
        recall = recall_score(test_labels, test_pred)
        f1 = f1_score(test_labels, test_pred)
        
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(test_labels, test_pred, target_names=['Background', 'Vehicle']))
        
        # Save the model
        model_path = 'vehicle_detector_model.h5'
        model.save(model_path)
        print(f"\n✓ Model saved to {model_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error in model training: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all processing steps"""
    print("\n" + "="*60)
    print("MELAUDIS Dataset Processing Pipeline")
    print("="*60)
    
    # Create necessary output directories
    os.makedirs(SPECTROGRAM_OUTPUT_PATH, exist_ok=True)

    steps = [
        ("Step 1: Audio Extraction (skipped)", step_1_extract_vehicle_audio),
        ("Step 2: Background Extraction (skipped)", step_2_extract_background_audio),
        ("Step 3: Spectrogram Generation (Train & Test)", step_3_generate_spectrograms),
        ("Step 4: Model Training & Evaluation", step_4_train_classifier),
    ]
    
    completed = []
    failed = []
    
    for step_name, step_func in steps:
        try:
            if step_func():
                completed.append(step_name)
            else:
                failed.append(step_name)
        except Exception as e:
            print(f"Fatal error in {step_name}: {e}")
            failed.append(step_name)
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Completed: {len(completed)}/{len(steps)}")
    if completed:
        for step in completed:
            print(f"  ✓ {step}")
    if failed:
        print(f"Failed: {len(failed)}")
        for step in failed:
            print(f"  ✗ {step}")
    
    if not failed:
        print("\n✓ All steps completed successfully!")
    else:
        print(f"\n✗ {len(failed)} step(s) failed. Please review the errors above.")

if __name__ == "__main__":
    main()
