
## **Step 6: Create Run Script**

# run_all.py
import subprocess
import sys
import time
import os

def run_step(step_name, command, check_output=True):
    """Run a step and print status"""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    
    try:
        if check_output:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        else:
            subprocess.run(command)
        
        print(f"✓ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {step_name} failed")
        print("Error:", e.stderr)
        return False
    except Exception as e:
        print(f"✗ {step_name} failed with exception")
        print("Exception:", str(e))
        return False

def main():
    """Run the complete XGBoost prediction pipeline"""
    
    print("\n" + "="*70)
    print("XGBOOST DEMAND PREDICTION PIPELINE")
    print("="*70)
    
    # Step 1: Install dependencies
    print("\nChecking dependencies...")
    try:
        import xgboost
        print("✓ XGBoost already installed")
    except ImportError:
        print("Installing XGBoost...")
        run_step("Install XGBoost", [sys.executable, "-m", "pip", "install", "xgboost"])
    
    # Step 2: Generate data
    run_step("Generate Enhanced Dataset", 
             [sys.executable, "generate_enhanced_data.py"])
    
    # Step 3: Run all models
    run_step("Run All Prediction Models", 
             [sys.executable, "models.py"])
    
    # Step 4: Create visualizations
    run_step("Create Model Comparisons", 
             [sys.executable, "visualize_comparison.py"])
    
    # Step 5: Check results
    if os.path.exists("all_model_predictions.json"):
        import json
        with open("all_model_predictions.json", "r") as f:
            data = json.load(f)
        
        print(f"\n{'='*60}")
        print("SUMMARY OF RESULTS")
        print(f"{'='*60}")
        
        for item, models in data.items():
            print(f"\n{item}:")
            for model_name, preds in models.items():
                if model_name == "XGBoost Metrics":
                    print(f"  {model_name}:")
                    for metric, value in preds.items():
                        print(f"    {metric}: {value}")
                elif model_name == "Top Features":
                    print(f"  {model_name} (Top 3):")
                    for feature, importance in preds[:3]:
                        print(f"    {feature}: {importance:.4f}")
                else:
                    if isinstance(preds, list):
                        avg_pred = sum(preds) / len(preds)
                        print(f"  {model_name}: Avg = {avg_pred:.1f}")
    
    # Step 6: Start API instructions
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("\nTo start the API server, run:")
    print("  python enhanced_api.py")
    print("\nThen open your browser to:")
    print("  http://localhost:8000")
    print("\nOr test with curl:")
    print("  curl http://localhost:8000/predict/Apple")
    print("  curl http://localhost:8000/compare/Banana")
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()