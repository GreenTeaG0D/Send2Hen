# Credit Card Default Prediction

A comprehensive machine learning pipeline for predicting credit card payment defaults using customer demographic, credit, and payment history data.
For 6001CMD - Machine Learning
By Bertram Emberton, 14492374

## Overview

This project implements a complete ML pipeline to predict whether a credit card client will default on their payment in the next month. The solution addresses a real-world problem relevant to insurance and risk management, helping financial institutions make informed decisions about credit risk.

**Dataset**: Default of Credit Card Clients from UCI Machine Learning Repository  
**Size**: ~30,000 records with 23 (starting) features  
**Task**: Binary classification (default vs. no default)
**Secondary Objective**: Model Comparison (LR v. RF v. GB v. MLP v. Stack(1-3))

## Project Structure

```
.
├── ingest.py                    # Data preprocessing and ingestion module
├── models.py                    # Model definitions and training functions
├── analysis.py                  # Model evaluation and analysis functions
├── ml_pipeline.ipynb           # Complete Jupyter notebook pipeline
├── default_of_credit_card_clients.xls  # Dataset file
├── results/                     # Generated visualizations and plots
│   ├── *_feature_importance.png
│   ├── *_shap_bar.png
│   └── *_shap_summary.png
├── plan.txt                    # Project planning document
└── README.md                   # This file
```

## Features

### Data Preprocessing (`ingest.py`)
- **Data Loading**: Reads Excel files and renames columns to descriptive names
- **Missing Value Handling**: 
  - Drops columns with >10% missing values
  - Uses K-NN imputation for numeric features
  - Mode imputation for categorical features
- **Anomaly Detection**: 
  - Z-score and IQR methods
  - Winsorization for outlier treatment
- **Feature Engineering**:
  - Aggregated payment features
  - Temporal features
  - Credit utilization ratios
- **Encoding**: 
  - One-hot encoding for nominal categories
  - Ordinal encoding for payment status
- **Scaling**: Standard or Robust scaling (configurable)
- **Class Imbalance Handling**: 
  - Stratified train/test splits
  - SMOTE (Synthetic Minority Oversampling Technique)
  - Class weights for imbalanced datasets

### Models (`models.py`)
Implements four different model types for comparison:

1. **Logistic Regression** (Linear)
   - L2 penalty
   - Balanced class weights for imbalanced data
   - Solvers: liblinear or saga

2. **Random Forest** (Tree-based)
   - Baseline ensemble method
   - 100 estimators

3. **Gradient Boosting** (Advanced Tree)
   - Built-in regularization
   - Handles heterogeneous features
   - Early stopping to prevent overfitting

4. **Neural Network** (Deep Learning)
   - Multi-layer Perceptron (MLP)
   - Hidden layers: (100, 50)
   - ReLU activation, Adam optimizer

**Model Training Features**:
- Stratified K-Fold Cross-Validation
- Hyperparameter tuning with RandomizedSearchCV and GridSearchCV
- Model calibration using CalibratedClassifierCV
- Feature importance analysis (permutation and SHAP)

### Analysis (`analysis.py`)
Comprehensive model evaluation including:

- **Performance Metrics**:
  - ROC-AUC (overall discrimination)
  - Precision, Recall, F1-Score
  - PR-AUC (better for class imbalance)
  - Brier Score (probabilistic prediction quality)
  - Confusion Matrix

- **Business Cost Analysis**:
  - Cost of false positives vs. false negatives
  - Total business cost calculation
  - Potential writeoff/payback analysis

- **Statistical Comparison**:
  - McNemar's test for model comparison
  - Paired t-test on cross-validation scores

- **Model Interpretability**:
  - Feature importance plots
  - SHAP (SHapley Additive exPlanations) values
  - SHAP summary and bar plots

## Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook (for running the notebook)

### Required Packages

```bash
pip install pandas numpy scikit-learn matplotlib scipy imbalanced-learn shap openpyxl
```

Or install individual packages:

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install scipy
pip install imbalanced-learn
pip install shap
pip install openpyxl  # For reading Excel files
```

## Usage

### Option 1: Using the Jupyter Notebook

1. Open `ml_pipeline.ipynb` in Jupyter Notebook
2. Run all cells to execute the complete pipeline
3. Results will be saved in the `results/` directory

### Option 2: Using Python Modules

#### Data Preprocessing

```python
from ingest import preprocess_data

# Preprocess data with train/test split
X_train, X_test, y_train, y_test, scaler = preprocess_data(
    file_path='default_of_credit_card_clients.xls',
    verbose=True,
    apply_scaling=True,
    split_data=True,
    apply_smote=True,
    test_size=0.2,
    random_state=42
)
```

#### Model Training

```python
from models import train_and_tune_models, get_logistic_regression, get_random_forest

# Train and tune multiple models
models_dict = train_and_tune_models(
    X_train, y_train, X_test, y_test,
    models_to_train=['logistic_regression', 'random_forest', 'gradient_boosting', 'neural_network'],
    tune_hyperparameters=True,
    random_state=42
)
```

#### Model Evaluation

```python
from analysis import comprehensive_evaluation, plot_feature_importance_and_shap

# Evaluate a model
results = comprehensive_evaluation(
    model=models_dict['Logistic Regression'],
    X_test=X_test,
    y_test=y_test,
    model_name='Logistic Regression',
    cost_false_positive=100,
    cost_false_negative=5000
)

# Generate feature importance and SHAP plots
plot_feature_importance_and_shap(
    model=models_dict['Logistic Regression'],
    X_train=X_train,
    X_test=X_test,
    feature_names=X_train.columns,
    model_name='Logistic Regression'
)
```

## Dataset

**Source**: UCI Machine Learning Repository  
**Citation**: Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H.

**Features**:
- `LIMIT_BAL`: Credit limit
- `SEX`: Gender (1=male, 2=female)
- `EDUCATION`: Education level
- `MARRIAGE`: Marital status
- `AGE`: Age in years
- `PAY_0` to `PAY_6`: Payment status for months -1 to -6
- `BILL_AMT1` to `BILL_AMT6`: Bill statement amounts
- `PAY_AMT1` to `PAY_AMT6`: Previous payment amounts
- `default_payment_next_month`: Target variable (1=default, 0=no default)

**Source**: ReadmeCodeGen FileTreeGenerator
**Citation**: https://readmecodegen.vercel.app/file-tree/file-tree-generator/github-project-tree-generator

**Features**:
- `Nice Tree`: generated the file tree above

## Results

The pipeline generates several outputs:

1. **Model Performance Metrics**: Printed to console during evaluation
2. **Feature Importance Plots**: Saved as PNG files in `results/`
3. **SHAP Visualizations**: 
   - SHAP summary plots
   - SHAP bar plots
4. **Statistical Comparisons**: Model comparison results

## Key Techniques

- **Stratified Sampling**: Maintains class distribution in train/test splits
- **SMOTE**: Synthetic oversampling to handle class imbalance
- **Cross-Validation**: Stratified K-Fold for robust model evaluation
- **Hyperparameter Tuning**: Randomized search followed by grid search
- **Model Calibration**: Probability calibration for reliable predictions
- **Feature Selection**: Permutation importance and SHAP for feature analysis
- **Statistical Testing**: McNemar's test and paired t-tests for model comparison

## Notes

- The dataset has significant class imbalance (~77% non-default, ~23% default)
- Tree-based models (Random Forest, Gradient Boosting) don't require feature scaling
- Linear models (Logistic Regression) and Neural Networks benefit from scaling
- SMOTE is applied only to training data to prevent data leakage
- All models use `random_state=42` for reproducibility

## Acknowledgments

- Dataset: UCI Machine Learning Repository
- Idea attribution: Vlad Login, SWE @ Admiral (for the real-world problem context)
- Tree Graphic: ReadmeCodeGen
