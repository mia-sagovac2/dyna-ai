#!/usr/bin/env python3
"""
Skripta za testiranje modela na testnom i validacijskom skupu, kao i na pojedinačnim slikama spektrograma. 
Dok jos tweakamo model treba pokretati samo na val skupu!!
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path


MODEL_PATH     = "../vehicle_detector_model.h5"
TEST_SPEC_PATH = "../data/spectrograms/test/"
VAL_SPEC_PATH  = "../data/spectrograms/val/"
IMAGE_SIZE     = (163, 223)


def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"!!Model not found at {model_path}")
        print("  Run 'python3 run_all.py' first to train the model.")
        return None
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
    return model

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    return np.array(img, dtype='float32') / 255.0

def predict_single(model, image_path):
    img = load_image(image_path)
    if img is None:
        print(f"!! Could not load image: {image_path}")
        return None
    prediction = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
    return {
        'label':      "Vehicle" if prediction > 0.5 else "Background",
        'confidence': float(abs(prediction - 0.5) * 2),
        'raw_score':  float(prediction),
        'image_path': image_path
    }

def evaluate_split(model, spec_path, split_name):
    """Run model on all spectrograms in a split and report accuracy"""
    print("\n" + "="*60)
    print(f"EVALUATING ON {split_name.upper()} SET")
    print("="*60)

    bg_path  = os.path.join(spec_path, 'BG')
    veh_path = os.path.join(spec_path, 'Veh')

    if not os.path.exists(bg_path) or not os.path.exists(veh_path):
        print(f"✗ Spectrogram directories not found at {spec_path}")
        return

    bg_results  = []
    veh_results = []

    print(f"\nProcessing background ({len(os.listdir(bg_path))} files)...")
    for file in os.listdir(bg_path):
        if file.endswith('.png'):
            result = predict_single(model, os.path.join(bg_path, file))
            if result:
                bg_results.append(result)

    print(f"Processing vehicles ({len(os.listdir(veh_path))} files)...")
    for file in os.listdir(veh_path):
        if file.endswith('.png'):
            result = predict_single(model, os.path.join(veh_path, file))
            if result:
                veh_results.append(result)

    # Accuracy
    bg_correct  = sum(1 for r in bg_results  if r['label'] == 'Background')
    veh_correct = sum(1 for r in veh_results if r['label'] == 'Vehicle')
    total       = len(bg_results) + len(veh_results)
    overall_acc = (bg_correct + veh_correct) / total * 100 if total > 0 else 0

    print(f"\n{'─'*40}")
    print(f"Background:  {bg_correct}/{len(bg_results)} correct "
          f"({bg_correct/len(bg_results)*100:.1f}%)" if bg_results else "Background: no files")
    print(f"Vehicle:     {veh_correct}/{len(veh_results)} correct "
          f"({veh_correct/len(veh_results)*100:.1f}%)" if veh_results else "Vehicle: no files")
    print(f"Overall:     {bg_correct + veh_correct}/{total} correct ({overall_acc:.1f}%)")

    if bg_results:
        print(f"\nAvg confidence - BG:  {np.mean([r['confidence'] for r in bg_results]):.3f}")
    if veh_results:
        print(f"Avg confidence - Veh: {np.mean([r['confidence'] for r in veh_results]):.3f}")

    # Sample predictions
    print(f"\nSample Background predictions (should be 'Background'):")
    for r in bg_results[:5]:
        status = "✓" if r['label'] == 'Background' else "✗"
        print(f"  {status} {r['label']} ({r['confidence']:.1%})")

    print(f"\nSample Vehicle predictions (should be 'Vehicle'):")
    for r in veh_results[:5]:
        status = "✓" if r['label'] == 'Vehicle' else "✗"
        print(f"  {status} {r['label']} ({r['confidence']:.1%})")


def test_custom_image(model, image_path):
    if not os.path.exists(image_path):
        print(f"!! Image not found: {image_path}")
        return
    print(f"\nTesting: {image_path}")
    result = predict_single(model, image_path)
    if result:
        print(f"  Prediction: {result['label']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Raw Score:  {result['raw_score']:.4f}")



def main():
    print("Test Suite")
    print("="*60)

    model = load_model(MODEL_PATH)
    if model is None:
        return

    print("\nModel Architecture:")
    model.summary()

    print("\n" + "="*60)
    print("TESTING OPTIONS")
    print("="*60)
    print("1. Evaluate on TEST set    ← finalno, tek kada smo gotovi sa tuningom modela")
    print("2. Evaluate on VAL set     ← koristi dok jos tweakamo model, da ne trošimo test set")
    print("3. Test on a single image")
    print("4. Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        evaluate_split(model, TEST_SPEC_PATH, "test")
    elif choice == "2":
        evaluate_split(model, VAL_SPEC_PATH, "val")
    elif choice == "3":
        image_path = input("Enter image path: ").strip()
        test_custom_image(model, image_path)
    elif choice == "4":
        print("Exiting...")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()