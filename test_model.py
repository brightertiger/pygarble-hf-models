#!/usr/bin/env python3
import sys
import os
import pandas as pd
from transformers import pipeline

def test_model(model_name="brightertiger/garbled-text-detector", num_samples=10):
    print(f"Testing model: {model_name}")
    print("=" * 70)
    
    try:
        print("\n1. Loading model...")
        classifier = pipeline(
            "text-classification", 
            model=model_name, 
            device=-1,
            truncation=True,
            max_length=512
        )
        print("✓ Model loaded successfully!")
        
        print("\n2. Loading test data from validation.csv...")
        
        if not os.path.exists("data/validation.csv"):
            print("✗ Error: data/validation.csv not found")
            print("\nPlease ensure validation.csv exists in the data/ directory")
            return 1
        
        df = pd.read_csv("data/validation.csv")
        print(f"✓ Loaded validation set with {len(df)} samples")
        
        samples_per_class = num_samples // 2
        df_normal = df[df['label'] == 0].sample(n=min(samples_per_class, len(df[df['label'] == 0])), random_state=42)
        df_garbled = df[df['label'] == 1].sample(n=min(samples_per_class, len(df[df['label'] == 1])), random_state=42)
        
        test_cases = []
        for _, row in df_normal.iterrows():
            test_cases.append({
                "text": row['text'],
                "expected": "NORMAL",
                "expected_label": 0
            })
        
        for _, row in df_garbled.iterrows():
            test_cases.append({
                "text": row['text'],
                "expected": "GARBLED",
                "expected_label": 1
            })
        
        print(f"✓ Selected {len(test_cases)} samples ({len(df_normal)} normal, {len(df_garbled)} garbled)")
        
        print("\n3. Running predictions...")
        
        passed = 0
        failed = 0
        predictions = []
        
        for i, test_case in enumerate(test_cases, 1):
            text = test_case["text"]
            expected = test_case["expected"]
            expected_label = test_case["expected_label"]
            
            result = classifier(text)[0]
            predicted = result["label"]
            confidence = result["score"]
            
            predicted_label = 1 if predicted == "GARBLED" else 0
            
            status = "✓ PASS" if predicted == expected else "✗ FAIL"
            
            if predicted == expected:
                passed += 1
            else:
                failed += 1
            
            predictions.append({
                "text": text[:100],
                "expected_label": expected_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "correct": predicted == expected
            })
            
            print(f"\n   Test {i}:")
            print(f"   Text: {text[:80]}{'...' if len(text) > 80 else ''}")
            print(f"   Expected: {expected}, Got: {predicted} (confidence: {confidence:.2%})")
            print(f"   {status}")
        
        print("\n" + "=" * 70)
        print(f"\n4. Test Results:")
        print(f"   Total: {len(test_cases)}")
        print(f"   Passed: {passed} ✓")
        print(f"   Failed: {failed} ✗")
        print(f"   Accuracy: {passed/len(test_cases):.1%}")
        
        if failed == 0:
            print(f"\n🎉 All tests passed! Model is working correctly.")
            return 0
        else:
            print(f"\n⚠️  {failed} test(s) failed. Please review the results above.")
            return 1
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"\nMake sure you have:")
        print(f"  1. Installed required packages: pip install transformers torch pandas")
        print(f"  2. Internet connection to download the model")
        print(f"  3. Sufficient disk space")
        return 1

def main():
    model_name = "brightertiger/garbled-text-detector"
    num_samples = 10
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    
    if len(sys.argv) > 2:
        try:
            num_samples = int(sys.argv[2])
        except ValueError:
            print(f"Warning: Invalid number of samples '{sys.argv[2]}', using default 10")
    
    print("Garbled Text Detector - Model Test")
    print("=" * 70)
    print(f"\nUsage: python test_model.py [model_name] [num_samples]")
    print(f"  model_name: HuggingFace model to test (default: {model_name})")
    print(f"  num_samples: Number of samples to test (default: 10)\n")
    
    exit_code = test_model(model_name, num_samples)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
