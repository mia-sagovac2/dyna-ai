#!/usr/bin/env python3
"""
Inference script: Classify new audio files using the trained model.
Converts WAV files to spectrograms and predicts whether they contain vehicles or background.
"""

import os
import sys
import cv2
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "../vehicle_detector_model.h5"
FRAME_SIZE = 512
HOP_SIZE = 64

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_model(model_path):
    """Load the trained model"""
    if not os.path.exists(model_path):
        print(f"✗ Model not found at {model_path}")
        print("  Run 'python3 run_all.py' first to train the model")
        return None
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    return model

def audio_to_spectrogram(audio_path):
    """Convert audio file to spectrogram image"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Convert to image
        plt.figure(figsize=(5, 3))
        librosa.display.specshow(log_mel, sr=sr, hop_length=HOP_SIZE, y_axis='mel')
        plt.axis('off')
        plt.savefig('temp_spec.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Load and preprocess image
        img = cv2.imread('temp_spec.png')
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (223, 163))
        img = np.array(img, dtype='float32') / 255.0
        
        # Clean up
        os.remove('temp_spec.png')
        
        return img
    except Exception as e:
        print(f"  Error processing {audio_path}: {e}")
        return None

def predict_audio(model, audio_path):
    """Predict if audio contains vehicle or background noise"""
    print(f"\nProcessing: {audio_path}")
    
    # Convert to spectrogram
    spec_img = audio_to_spectrogram(audio_path)
    if spec_img is None:
        return None
    
    # Add batch dimension
    spec_batch = np.expand_dims(spec_img, axis=0)
    
    # Make prediction
    prediction = model.predict(spec_batch, verbose=0)[0][0]
    confidence = abs(prediction - 0.5) * 2
    
    label = "🚗 VEHICLE" if prediction > 0.5 else "🔇 BACKGROUND"
    
    return {
        'label': label,
        'confidence': float(confidence),
        'raw_score': float(prediction),
        'audio_path': audio_path
    }

def batch_predict_directory(model, directory_path):
    """Predict on all audio files in a directory"""
    if not os.path.isdir(directory_path):
        print(f"✗ Directory not found: {directory_path}")
        return
    
    print(f"\nProcessing all .wav files in {directory_path}...")
    results = []
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.wav'):
                audio_path = os.path.join(root, file)
                result = predict_audio(model, audio_path)
                if result:
                    results.append(result)
    
    # Summary
    if results:
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        
        vehicles = sum(1 for r in results if 'VEHICLE' in r['label'])
        background = sum(1 for r in results if 'BACKGROUND' in r['label'])
        
        print(f"Total files processed: {len(results)}")
        print(f"  🚗 Vehicles: {vehicles}")
        print(f"  🔇 Background: {background}")
        print(f"\nSample predictions:")
        for result in results[:10]:
            print(f"  {result['label']} ({result['confidence']:.1%}) - {result['audio_path']}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("MELAUDIS Vehicle Detection - Audio Inference")
    print("="*60)
    
    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        return
    
    print("✓ Model loaded successfully\n")
    
    # Choose inference mode
    print("OPTIONS:")
    print("1. Classify a single audio file")
    print("2. Classify all audio files in a directory")
    print("3. Classify all files in data/vehicles/")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        audio_path = input("Enter audio file path (.wav): ").strip()
        result = predict_audio(model, audio_path)
        if result:
            print(f"\n{'='*60}")
            print("RESULT")
            print(f"{'='*60}")
            print(f"Prediction: {result['label']}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Raw Score: {result['raw_score']:.4f}")
    
    elif choice == "2":
        dir_path = input("Enter directory path: ").strip()
        batch_predict_directory(model, dir_path)
    
    elif choice == "3":
        batch_predict_directory(model, "./data/raw/vehicles/")
    
    elif choice == "4":
        print("Exiting...")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
