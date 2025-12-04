"""
Main script to run the complete ML pipeline:
1. Data Ingestion and Preprocessing (ingest.py)
2. Model Training and Tuning (models.py)
3. Model Analysis and Evaluation (analysis.py)
"""

import os
import sys
from ingest import preprocess_data
from models import train_and_tune_models, create_stacked_model
from analysis import (
    comprehensive_evaluation,
    plot_feature_importance_and_shap,
    cross_validate_models,
    statistical_comparison
)

def main():
    """
    Main function to run the complete ML pipeline.
    """

    print("Credit Card Default Prediction - ML Pipeline")
    
    # Configuration
    DATA_FILE = 'default_of_credit_card_clients.xls'
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VERBOSE = True
    
    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found!")
        print("Please ensure the data file is in the current directory.")
        sys.exit(1)
    
    # ============================================================================
    # STEP 1: DATA INGESTION AND PREPROCESSING
    # ============================================================================
    print("STEP 1: DATA INGESTION AND PREPROCESSING")
    
    try:
        X_train, X_test, y_train, y_test, scaler = preprocess_data(
            file_path=DATA_FILE,
            verbose=VERBOSE,
            apply_scaling=True,
            split_data=True,
            apply_smote=True,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        
        print(f"\n✓ Data preprocessing completed successfully!")
        print(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"  Scaler: {type(scaler).__name__ if scaler else 'None'}")
        
    except Exception as e:
        print(f"\n✗ Error during data preprocessing: {e}")
        sys.exit(1)
    
    # ============================================================================
    # STEP 2: MODEL TRAINING AND TUNING
    # ============================================================================
    print("STEP 2: MODEL TRAINING AND TUNING")
    
    try:
        models_dict, training_results = train_and_tune_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            use_randomized=True,
            use_grid=True,
            n_iter_randomized=50,
            cv=5,
            random_state=RANDOM_STATE,
            verbose=VERBOSE
        )
        
        print(f"\n✓ Model training completed successfully!")
        print(f"  Trained {len(models_dict)} models:")
        for model_name in models_dict.keys():
            print(f"    - {model_name}")
        
        # Create stacked ensemble model
        print("Creating Stacked Ensemble Model...")
        
        try:
            stacked_model = create_stacked_model(
                models_dict=models_dict,
                X_train=X_train,
                y_train=y_train,
                random_state=RANDOM_STATE,
                calibrate=False
            )
            models_dict['stacked_ensemble'] = stacked_model
            print("✓ Stacked ensemble model created successfully!")
        except Exception as e:
            print(f"⚠ Warning: Could not create stacked ensemble: {e}")
            print("  Continuing without stacked ensemble...")
        
    except Exception as e:
        print(f"\n✗ Error during model training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ============================================================================
    # STEP 3: MODEL ANALYSIS AND EVALUATION
    # ============================================================================
    print("STEP 3: MODEL ANALYSIS AND EVALUATION")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Model name mapping for display
    model_display_names = {
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'gradient_boosting': 'Gradient Boosting',
        'neural_network': 'Neural Network',
        'stacked_ensemble': 'Stacked Ensemble'
    }
    
    # Comprehensive evaluation for each model
    print("\n" + "-" * 80)
    print("Comprehensive Model Evaluation")
    print("-" * 80)
    
    evaluation_results = {}
    for model_key, model in models_dict.items():
        model_name = model_display_names.get(model_key, model_key)
        print(f"\nEvaluating {model_name}...")
        
        try:
            eval_results = comprehensive_evaluation(
                model=model,
                X_test=X_test,
                y_test=y_test,
                model_name=model_name,
                cost_false_positive=100,
                cost_false_negative=5000,
                verbose=VERBOSE
            )
            evaluation_results[model_key] = eval_results
        except Exception as e:
            print(f"⚠ Warning: Error evaluating {model_name}: {e}")
            continue
    
    # Cross-validation
    print("Cross-Validation Analysis")
    
    try:
        cv_results = cross_validate_models(
            models_dict=models_dict,
            X_train=X_train,
            y_train=y_train,
            cv=5,
            random_state=RANDOM_STATE,
            verbose=VERBOSE
        )
    except Exception as e:
        print(f"⚠ Warning: Error during cross-validation: {e}")
        cv_results = None
    
    # Statistical comparison
    print("Statistical Model Comparison")
    
    try:
        stat_comparison = statistical_comparison(
            models_dict=models_dict,
            X_test=X_test,
            y_test=y_test,
            cv_results=cv_results,
            verbose=VERBOSE
        )
    except Exception as e:
        print(f"⚠ Warning: Error during statistical comparison: {e}")
    
    # Feature importance and SHAP plots
    print("Generating Feature Importance and SHAP Plots")
    
    feature_names = X_train.columns.tolist()
    
    for model_key, model in models_dict.items():
        model_name = model_display_names.get(model_key, model_key)
        print(f"\nGenerating plots for {model_name}...")
        
        try:
            plot_feature_importance_and_shap(
                model=model,
                X_train=X_train,
                X_test=X_test,
                feature_names=feature_names,
                model_name=model_name,
                top_n=20,
                save_plots=True,
                y_test=y_test
            )
            print(f"✓ Plots generated for {model_name}")
        except Exception as e:
            print(f"⚠ Warning: Error generating plots for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("PIPELINE EXECUTION COMPLETE")
    
    print("\nSummary:")
    print(f"Data preprocessing: Completed")
    print(f"Models trained: {len(models_dict)}")
    print(f"Evaluations performed: {len(evaluation_results)}")
    print(f"Results saved to: results/")
    
    # Find best model by ROC-AUC
    if evaluation_results:
        best_model_key = None
        best_roc_auc = 0
        
        for model_key, results in evaluation_results.items():
            if 'roc_auc' in results and results['roc_auc'] > best_roc_auc:
                best_roc_auc = results['roc_auc']
                best_model_key = model_key
        
        if best_model_key:
            best_model_name = model_display_names.get(best_model_key, best_model_key)
            print(f"\n  Best Model (by ROC-AUC): {best_model_name} ({best_roc_auc:.4f})")
    
    print("Pipeline execution finished successfully!")

if __name__ == "__main__":
    main()

