#!/usr/bin/env python3
"""
Test the trained vehicle detection model on spectrogram images.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "../vehicle_detector_model.h5"
SPECTROGRAM_PATH = "../data/spectrograms/test/"
TEST_IMAGE_SIZE = (163, 223, 3)

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
    print("✓ Model loaded successfully")
    return model

def load_image(image_path):
    """Load and preprocess an image"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (223, 163))
    img = np.array(img, dtype='float32') / 255.0
    return img

def test_single_image(model, image_path):
    """Test model on a single image"""
    img = load_image(image_path)
    if img is None:
        print(f"✗ Could not load image: {image_path}")
        return None
    
    # Add batch dimension
    img_batch = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img_batch, verbose=0)[0][0]
    confidence = abs(prediction - 0.5) * 2
    
    label = "Vehicle" if prediction > 0.5 else "Background"
    return {
        'label': label,
        'confidence': float(confidence),
        'raw_score': float(prediction),
        'image_path': image_path
    }

def test_all_spectrograms(model):
    """Test model on all spectrogram images"""
    print("\n" + "="*60)
    print("TESTING ON ALL SPECTROGRAMS")
    print("="*60)
    
    results = {'vehicle': [], 'background': []}
    total = 0
    
    # Test BG spectrograms
    bg_path = os.path.join(SPECTROGRAM_PATH, 'BG')
    if os.path.exists(bg_path):
        print(f"\nTesting background spectrograms ({len(os.listdir(bg_path))} files)...")
        for i, file in enumerate(os.listdir(bg_path)):
            if file.endswith('.png'):
                img_path = os.path.join(bg_path, file)
                result = test_single_image(model, img_path)
                if result:
                    results['background'].append(result)
                    total += 1
                    if (i + 1) % 100 == 0:
                        print(f"  Processed {i + 1} background images...")
    
    # Test Veh spectrograms
    veh_path = os.path.join(SPECTROGRAM_PATH, 'Veh')
    if os.path.exists(veh_path):
        print(f"\nTesting vehicle spectrograms ({len(os.listdir(veh_path))} files)...")
        for i, file in enumerate(os.listdir(veh_path)):
            if file.endswith('.png'):
                img_path = os.path.join(veh_path, file)
                result = test_single_image(model, img_path)
                if result:
                    results['vehicle'].append(result)
                    total += 1
                    if (i + 1) % 100 == 0:
                        print(f"  Processed {i + 1} vehicle images...")
    
    return results, total

def calculate_accuracy(results):
    """Calculate accuracy metrics"""
    print("\n" + "="*60)
    print("ACCURACY METRICS")
    print("="*60)
    
    bg_correct = sum(1 for r in results['background'] if r['label'] == 'Background')
    bg_total = len(results['background'])
    bg_accuracy = (bg_correct / bg_total * 100) if bg_total > 0 else 0
    
    veh_correct = sum(1 for r in results['vehicle'] if r['label'] == 'Vehicle')
    veh_total = len(results['vehicle'])
    veh_accuracy = (veh_correct / veh_total * 100) if veh_total > 0 else 0
    
    overall_correct = bg_correct + veh_correct
    overall_total = bg_total + veh_total
    overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0
    
    print(f"\nBackground Classification:")
    print(f"  Correct: {bg_correct}/{bg_total}")
    print(f"  Accuracy: {bg_accuracy:.1f}%")
    print(f"  Avg Confidence: {np.mean([r['confidence'] for r in results['background']]):.3f}")
    
    print(f"\nVehicle Classification:")
    print(f"  Correct: {veh_correct}/{veh_total}")
    print(f"  Accuracy: {veh_accuracy:.1f}%")
    print(f"  Avg Confidence: {np.mean([r['confidence'] for r in results['vehicle']]):.3f}")
    
    print(f"\nOverall:")
    print(f"  Correct: {overall_correct}/{overall_total}")
    print(f"  Accuracy: {overall_accuracy:.1f}%")
    
    return {
        'bg_accuracy': bg_accuracy,
        'veh_accuracy': veh_accuracy,
        'overall_accuracy': overall_accuracy
    }

def test_custom_image(model, image_path):
    """Test model on a custom image"""
    if not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        return
    
    print(f"\nTesting custom image: {image_path}")
    result = test_single_image(model, image_path)
    if result:
        print(f"  Prediction: {result['label']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Raw Score: {result['raw_score']:.4f}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("MELAUDIS Vehicle Detection Model - Test Suite")
    print("="*60)
    
    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        return
    
    print("\nModel Architecture:")
    model.summary()
    
    # Choose testing mode
    print("\n" + "="*60)
    print("TESTING OPTIONS")
    print("="*60)
    print("1. Test on all spectrograms (class-wise accuracy)")
    print("2. Test on single image")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        results, total = test_all_spectrograms(model)
        if total > 0:
            calculate_accuracy(results)
            
            # Show some predictions
            print("\n" + "="*60)
            print("SAMPLE PREDICTIONS")
            print("="*60)
            print("\nBackground samples (should predict as 'Background'):")
            for result in results['background'][:5]:
                status = "✓" if result['label'] == 'Background' else "✗"
                print(f"  {status} {result['label']} ({result['confidence']:.1%})")
            
            print("\nVehicle samples (should predict as 'Vehicle'):")
            for result in results['vehicle'][:5]:
                status = "✓" if result['label'] == 'Vehicle' else "✗"
                print(f"  {status} {result['label']} ({result['confidence']:.1%})")
    
    elif choice == "2":
        image_path = input("Enter image path: ").strip()
        test_custom_image(model, image_path)
    
    elif choice == "3":
        print("Exiting...")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
