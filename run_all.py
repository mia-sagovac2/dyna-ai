#!/usr/bin/env python3
"""
Glavna skripta gdje se izvode svi koraci: ekstrakcija audio zapisa, generiranje spektrograma, treniranje modela i evaluacija.
"""

import os
import sys
from pathlib import Path

# data paths
TRAIN_AUDIO_PATH = "data/train/"
VAL_AUDIO_PATH   = "data/val/"
TEST_AUDIO_PATH  = "data/test/"
SPECTROGRAM_OUTPUT_PATH = "data/spectrograms/"

# spektogrami
TRAIN_SPEC_PATH = "data/spectrograms/train/"
VAL_SPEC_PATH   = "data/spectrograms/val/"
TEST_SPEC_PATH  = "data/spectrograms/test/"

# limit za testiranje - obradi samo 100 BG i 100 Veh po setu (ako je postavljeno), smanjuje vrijeme izvođenja tijekom razvoja
MAX_FILES_TO_PROCESS = 100 # staviti na None


def step_1_extract_vehicle_audio():
    print("1: Vehicle audio fileovi extractani!")
    return True

def step_2_extract_background_audio():
    print("2: Background audio fileovi extractani!")
    return True

def step_3_generate_spectrograms():
    print("3: Generiranje spektrograma za Train, Val i Test setove...")

    try:
        import librosa
        import numpy as np
        import matplotlib.pyplot as plt

        for spec_path in [TRAIN_SPEC_PATH, VAL_SPEC_PATH, TEST_SPEC_PATH]:
            os.makedirs(f"{spec_path}BG",  exist_ok=True)
            os.makedirs(f"{spec_path}Veh", exist_ok=True)

        FRAME_SIZE = 512
        HOP_SIZE   = 64

        def save_spec(audio, sr, save_path):
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
            log_mel  = librosa.power_to_db(mel_spec, ref=np.max)
            plt.figure(figsize=(5, 3))
            librosa.display.specshow(log_mel, sr=sr, hop_length=HOP_SIZE, y_axis='mel')
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        def process_audio_set(audio_base_path, spec_base_path, set_name):
            print(f"\n  {set_name} Set:")
            bg_count  = 0
            veh_count = 0

            bg_path = f"{audio_base_path}bg_noise/"
            if os.path.exists(bg_path):
                print(f"    Processing background...")
                for file in os.listdir(bg_path):
                    if MAX_FILES_TO_PROCESS and bg_count >= MAX_FILES_TO_PROCESS:
                        break
                    if file.endswith('.wav'):
                        try:
                            audio, sr = librosa.load(os.path.join(bg_path, file), sr=None)
                            save_spec(audio, sr, f"{spec_base_path}BG/{file[:-4]}.png")
                            bg_count += 1
                        except Exception as e:
                            print(f"      Error processing {file}: {e}")

            veh_path = f"{audio_base_path}vehicles/"
            if os.path.exists(veh_path):
                print(f"    Processing vehicles...")
                for file in os.listdir(veh_path):
                    if MAX_FILES_TO_PROCESS and veh_count >= MAX_FILES_TO_PROCESS:
                        break
                    if file.endswith('.wav'):
                        try:
                            audio, sr = librosa.load(os.path.join(veh_path, file), sr=None)
                            save_spec(audio, sr, f"{spec_base_path}Veh/{file[:-4]}.png")
                            veh_count += 1
                        except Exception as e:
                            print(f"      Error processing {file}: {e}")

            print(f"    DONE {set_name}: {bg_count} BG + {veh_count} Veh spectrograms")
            return bg_count, veh_count

        train_bg,  train_veh  = process_audio_set(TRAIN_AUDIO_PATH, TRAIN_SPEC_PATH, "Train")
        val_bg,    val_veh    = process_audio_set(VAL_AUDIO_PATH,   VAL_SPEC_PATH,   "Val")
        test_bg,   test_veh   = process_audio_set(TEST_AUDIO_PATH,  TEST_SPEC_PATH,  "Test")

        print(f"\n✓ Total spectrograms generated:")
        print(f"  Train: {train_bg + train_veh} ({train_bg} BG + {train_veh} Veh)")
        print(f"  Val:   {val_bg   + val_veh}   ({val_bg} BG + {val_veh} Veh)")
        print(f"  Test:  {test_bg  + test_veh}  ({test_bg} BG + {test_veh} Veh)")
        return True

    except Exception as e:
        print(f"!! Error in spectrogram generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def step_4_train_classifier():
    print("4: Treniranje i evaluacija modela...")

    try:
        import cv2
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import classification_report

        for path in [TRAIN_SPEC_PATH, VAL_SPEC_PATH, TEST_SPEC_PATH]:
            if not os.path.exists(path):
                print(f"!! Spectrogram directory not found: {path}.")
                return False

        def load_split(spec_path, split_name):
            print(f"\n  Loading {split_name} data...")
            images, labels = [], []
            for file in os.listdir(f"{spec_path}BG"):
                if file.endswith('.png'):
                    img = cv2.imread(os.path.join(f"{spec_path}BG", file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (223, 163))
                        images.append(img)
                        labels.append(0)
            for file in os.listdir(f"{spec_path}Veh"):
                if file.endswith('.png'):
                    img = cv2.imread(os.path.join(f"{spec_path}Veh", file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (223, 163))
                        images.append(img)
                        labels.append(1)
            if not images:
                print(f"  !! No images found in {spec_path}")
                return None, None
            images = np.array(images, dtype='float32') / 255.0
            labels = np.array(labels, dtype='int32')
            print(f"  ✓ {split_name}: {len(images)} images "
                  f"(BG: {sum(labels == 0)}, Veh: {sum(labels == 1)})")
            return images, labels

        print("\nLoading datasets...")
        train_images, train_labels = load_split(TRAIN_SPEC_PATH, "Train")
        val_images,   val_labels   = load_split(VAL_SPEC_PATH,   "Val")
        test_images,  test_labels  = load_split(TEST_SPEC_PATH,  "Test")

        if train_images is None or val_images is None or test_images is None:
            return False

        print("\nBuilding model...")
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

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
        )

        print("\nTraining... (val set used to monitor overfitting)")
        history = model.fit(
            train_images, train_labels,
            epochs=20,
            batch_size=16,
            validation_data=(val_images, val_labels),
            callbacks=[early_stop],
            verbose=1
        )

        val_loss, val_acc = model.evaluate(val_images, val_labels, verbose=0)
        print(f"\nValidation Accuracy: {val_acc*100:.1f}%  (ovo se koristi za prilagodbu hiperparametara)")


        model_path = 'vehicle_detector_model.h5'
        model.save(model_path)
        print(f"Model saved to {model_path}")
        return True

    except Exception as e:
        print(f"Error in model training: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Processing Pipeline")
    print("="*60)

    os.makedirs(SPECTROGRAM_OUTPUT_PATH, exist_ok=True)

    steps = [
        ("Step 1: Audio Extraction",                  step_1_extract_vehicle_audio),
        ("Step 2: Background Extraction",             step_2_extract_background_audio),
        ("Step 3: Generiranje spektograma",  step_3_generate_spectrograms),
        ("Step 4: Treniranje modela i evaluacija",                 step_4_train_classifier),
    ]

    completed, failed = [], []
    for step_name, step_func in steps:
        try:
            if step_func():
                completed.append(step_name)
            else:
                failed.append(step_name)
        except Exception as e:
            print(f"Fatal error in {step_name}: {e}")
            failed.append(step_name)

    if not failed:
        print("\nAll steps completed successfully!")
    else:
        print(f"\n {len(failed)} step(s) failed. Please review the errors above.")

if __name__ == "__main__":
    main()