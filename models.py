import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix, accuracy_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

def check_class_imbalance(y):
	"""
	Check if class imbalance is significant (>60:40 ratio).
	
	Returns:
		bool: True if imbalance is significant
	"""
	class_counts = y.value_counts()
	ratio = class_counts.min() / class_counts.max()
	return ratio < 0.4

def get_logistic_regression(class_imbalance=False, random_state=42):
	"""
	Create Logistic Regression model with L2 penalty.
	
	Args:
		class_imbalance: If True, use class_weight='balanced'
		random_state: Random seed
	
	Returns:
		LogisticRegression model
	"""
	if class_imbalance:
		return LogisticRegression(
			penalty='l2',
			solver='liblinear',
			class_weight='balanced',
			random_state=random_state,
			max_iter=1000
		)
	else:
		return LogisticRegression(
			penalty='l2',
			solver='saga',
			random_state=random_state,
			max_iter=1000
		)

def get_random_forest(random_state=42):
	"""
	Create Random Forest classifier as baseline.
	
	Returns:
		RandomForestClassifier model
	"""
	return RandomForestClassifier(
		n_estimators=100,
		random_state=random_state,
		n_jobs=-1
	)

def get_gradient_boosting(random_state=42):
	"""
	Create Gradient Boosting Tree classifier.
	
	Returns:
		GradientBoostingClassifier model
	"""
	return GradientBoostingClassifier(
		random_state=random_state,
		validation_fraction=0.1,
		n_iter_no_change=5
	)

def get_neural_network(random_state=42):
	"""
	Create simple Neural Network (MLP) for comparison.
	
	Returns:
		MLPClassifier model
	"""
	return MLPClassifier(
		hidden_layer_sizes=(100, 50),
		activation='relu',
		solver='adam',
		alpha=0.0001,
		batch_size='auto',
		learning_rate='constant',
		learning_rate_init=0.001,
		max_iter=500,
		random_state=random_state,
		early_stopping=True,
		validation_fraction=0.1
	)

def tune_model_randomized(model, param_distributions, X_train, y_train, 
						  n_iter=50, cv_fold=None, scoring='roc_auc', random_state=42, verbose=0):
	"""
	Use RandomizedSearchCV for broad hyperparameter search.
	
	Args:
		model: Base model to tune
		param_distributions: Dictionary of parameter distributions
		X_train: Training features
		y_train: Training target
		n_iter: Number of iterations
		cv_fold: Pre-defined CV fold object (for fairness across models)
		scoring: Scoring metric
		random_state: Random seed
		verbose: Verbosity level
	
	Returns:
		Best model from RandomizedSearchCV
	"""
	if cv_fold is None:
		cv_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
	
	random_search = RandomizedSearchCV(
		estimator=model,
		param_distributions=param_distributions,
		n_iter=n_iter,
		cv=cv_fold,
		scoring=scoring,
		random_state=random_state,
		n_jobs=-1,
		verbose=verbose
	)
	
	random_search.fit(X_train, y_train)
	return random_search.best_estimator_, random_search.best_params_, random_search.best_score_

def tune_model_grid(model, param_grid, X_train, y_train, 
					cv_fold=None, scoring='roc_auc', verbose=0):
	"""
	Use GridSearchCV to refine hyperparameter search in target region.
	
	Args:
		model: Base model to tune
		param_grid: Dictionary of parameter grid
		X_train: Training features
		y_train: Training target
		cv_fold: Pre-defined CV fold object (for fairness across models)
		scoring: Scoring metric
		verbose: Verbosity level
	
	Returns:
		Best model from GridSearchCV
	"""
	if cv_fold is None:
		cv_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	
	grid_search = GridSearchCV(
		estimator=model,
		param_grid=param_grid,
		cv=cv_fold,
		scoring=scoring,
		n_jobs=-1,
		verbose=verbose
	)
	
	grid_search.fit(X_train, y_train)
	return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model", verbose=True):
	"""
	Evaluate model using ROC-AUC and F1 score.
	
	Args:
		model: Trained model
		X_train: Training features
		y_train: Training target
		X_test: Test features
		y_test: Test target
		model_name: Name for logging
		verbose: If True, print results
	
	Returns:
		Dictionary with evaluation metrics
	"""
	# Predictions
	y_train_pred = model.predict(X_train)
	y_test_pred = model.predict(X_test)
	
	# Probabilities for ROC-AUC
	y_train_proba = model.predict_proba(X_train)[:, 1]
	y_test_proba = model.predict_proba(X_test)[:, 1]
	
	# Calculate metrics
	train_roc_auc = roc_auc_score(y_train, y_train_proba)
	test_roc_auc = roc_auc_score(y_test, y_test_proba)
	train_f1 = f1_score(y_train, y_train_pred)
	test_f1 = f1_score(y_test, y_test_pred)
	
	results = {
		'train_roc_auc': train_roc_auc,
		'test_roc_auc': test_roc_auc,
		'train_f1': train_f1,
		'test_f1': test_f1,
		'confusion_matrix': confusion_matrix(y_test, y_test_pred),
		'classification_report': classification_report(y_test, y_test_pred)
	}
	
	if verbose:
		print(f"\n{model_name} Evaluation:")
		print(f"Train ROC-AUC: {train_roc_auc:.4f}")
		print(f"Test ROC-AUC: {test_roc_auc:.4f}")
		print(f"Train F1: {train_f1:.4f}")
		print(f"Test F1: {test_f1:.4f}")
		print(f"\nConfusion Matrix:\n{results['confusion_matrix']}")
		print(f"\nClassification Report:\n{results['classification_report']}")
	
	return results

def get_hyperparameter_grids():
	"""
	Define hyperparameter grids for each model.
	
    # Parameter ranges based on scikit-learn docs defaults and common practice from Hastie et al. (2009)
	# Expanded ranges for randomized search, narrower grids for refinement

	Returns:
		Dictionary of parameter distributions and grids for each model
	"""
	grids = {
		'logistic_regression': {
			'randomized': {
				'C': np.logspace(-4, 4, 20),
				'solver': ['liblinear', 'saga'],
				'max_iter': [500, 1000, 2000]
			},
			'grid': {
				'C': np.logspace(-2, 2, 10),
				'solver': ['liblinear', 'saga']
			}
		},
		'random_forest': {
			'randomized': {
				'n_estimators': [50, 100, 200, 300],
				'max_depth': [None, 10, 20, 30],
				'min_samples_split': [2, 5, 10],
				'min_samples_leaf': [1, 2, 4],
				'max_features': ['sqrt', 'log2', None]
			},
			'grid': {
				'n_estimators': [100, 200],
				'max_depth': [10, 20, None],
				'min_samples_split': [2, 5],
				'max_features': ['sqrt', 'log2']
			}
		},
		'gradient_boosting': {
			'randomized': {
				'n_estimators': [50, 100, 200],
				'learning_rate': [0.01, 0.1, 0.2],
				'max_depth': [3, 5, 7],
				'min_samples_split': [2, 5],
				'min_samples_leaf': [1, 2],
				'subsample': [0.8, 0.9, 1.0]
			},
			'grid': {
				'n_estimators': [100, 200],
				'learning_rate': [0.05, 0.1],
				'max_depth': [3, 5],
				'subsample': [0.8, 0.9]
			}
		},
		'neural_network': {
			'randomized': {
				'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100)],
				'alpha': [0.0001, 0.001, 0.01],
				'learning_rate_init': [0.001, 0.01],
				'max_iter': [300, 500]
			},
			'grid': {
				'hidden_layer_sizes': [(100,), (100, 50)],
				'alpha': [0.0001, 0.001],
				'learning_rate_init': [0.001, 0.01]
			}
		}
	}
	return grids

def train_and_tune_models(X_train, y_train, X_test, y_test, 
						  use_randomized=True, use_grid=True, 
						  n_iter_randomized=50, cv=5, random_state=42, verbose=True):
	"""
	Train and tune all models using RandomizedSearchCV and GridSearchCV.
	
	Args:
		X_train: Training features
		y_train: Training target
		X_test: Test features
		y_test: Test target
		use_randomized: If True, use RandomizedSearchCV first
		use_grid: If True, refine with GridSearchCV
		n_iter_randomized: Number of iterations for RandomizedSearchCV
		cv: Number of CV folds
		random_state: Random seed
		verbose: If True, print progress
	
	Returns:
		Dictionary of trained models and their results
	"""
	models = {}
	results = {}
	grids = get_hyperparameter_grids()
	total_start_time = time.time()
	
	cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
	
	class_imbalance = check_class_imbalance(y_train)
	if verbose:
		print(f"Class imbalance detected: {class_imbalance}")
	
	model_start_time = time.time()
	if verbose:
		print("\nTraining Logistic Regression...")
	lr = get_logistic_regression(class_imbalance=class_imbalance, random_state=random_state)
	
	if use_randomized:
		lr_best, lr_params, lr_score = tune_model_randomized(
			lr, grids['logistic_regression']['randomized'],
			X_train, y_train, n_iter=n_iter_randomized, cv_fold=cv_fold, random_state=random_state, verbose=0
		)
		if verbose:
			print(f"RandomizedSearchCV best score: {lr_score:.4f}")
			print(f"Best params: {lr_params}")
		
		if use_grid:
			# Refine with grid search around best params
			grid_params = grids['logistic_regression']['grid'].copy()
			if 'C' in lr_params:
				best_c = lr_params['C']
				# Narrow C range around best value
				grid_params['C'] = np.logspace(np.log10(best_c * 0.5), np.log10(best_c * 2), 5)
			
			lr_best, lr_params, lr_score = tune_model_grid(
				lr_best, grid_params, X_train, y_train, cv_fold=cv_fold, verbose=0
			)
			if verbose:
				print(f"GridSearchCV best score: {lr_score:.4f}")
				print(f"Refined params: {lr_params}")
	else:
		lr_best = lr
		lr_best.fit(X_train, y_train)
	
	models['logistic_regression'] = lr_best
	results['logistic_regression'] = evaluate_model(
		lr_best, X_train, y_train, X_test, y_test, "Logistic Regression", verbose=verbose
	)
	model_time = time.time() - model_start_time
	accuracy = accuracy_score(y_test, lr_best.predict(X_test))
	print(f"Logistic Regression finished, training time: {model_time:.2f}s, accuracy: {accuracy:.4f}")
	
	model_start_time = time.time()
	if verbose:
		print("\nTraining Random Forest...")
	rf = get_random_forest(random_state=random_state)
	
	if use_randomized:
		rf_best, rf_params, rf_score = tune_model_randomized(
			rf, grids['random_forest']['randomized'],
			X_train, y_train, n_iter=n_iter_randomized, cv_fold=cv_fold, random_state=random_state, verbose=0
		)
		if verbose:
			print(f"RandomizedSearchCV best score: {rf_score:.4f}")
			print(f"Best params: {rf_params}")
		
		if use_grid:
			rf_best, rf_params, rf_score = tune_model_grid(
				rf_best, grids['random_forest']['grid'], X_train, y_train, cv_fold=cv_fold, verbose=0
			)
			if verbose:
				print(f"GridSearchCV best score: {rf_score:.4f}")
				print(f"Refined params: {rf_params}")
	else:
		rf_best = rf
		rf_best.fit(X_train, y_train)
	
	models['random_forest'] = rf_best
	results['random_forest'] = evaluate_model(
		rf_best, X_train, y_train, X_test, y_test, "Random Forest", verbose=verbose
	)
	model_time = time.time() - model_start_time
	accuracy = accuracy_score(y_test, rf_best.predict(X_test))
	print(f"Random Forest finished, training time: {model_time:.2f}s, accuracy: {accuracy:.4f}")
	
	model_start_time = time.time()
	if verbose:
		print("\nTraining Gradient Boosting...")
	gb = get_gradient_boosting(random_state=random_state)
	
	if use_randomized:
		gb_best, gb_params, gb_score = tune_model_randomized(
			gb, grids['gradient_boosting']['randomized'],
			X_train, y_train, n_iter=n_iter_randomized, cv_fold=cv_fold, random_state=random_state, verbose=0
		)
		if verbose:
			print(f"RandomizedSearchCV best score: {gb_score:.4f}")
			print(f"Best params: {gb_params}")
		
		if use_grid:
			gb_best, gb_params, gb_score = tune_model_grid(
				gb_best, grids['gradient_boosting']['grid'], X_train, y_train, cv_fold=cv_fold, verbose=0
			)
			if verbose:
				print(f"GridSearchCV best score: {gb_score:.4f}")
				print(f"Refined params: {gb_params}")
	else:
		gb_best = gb
		gb_best.fit(X_train, y_train)
	
	models['gradient_boosting'] = gb_best
	results['gradient_boosting'] = evaluate_model(
		gb_best, X_train, y_train, X_test, y_test, "Gradient Boosting", verbose=verbose
	)
	model_time = time.time() - model_start_time
	accuracy = accuracy_score(y_test, gb_best.predict(X_test))
	print(f"Gradient Boosting finished, training time: {model_time:.2f}s, accuracy: {accuracy:.4f}")
	
	model_start_time = time.time()
	if verbose:
		print("\nTraining Neural Network...")
	nn = get_neural_network(random_state=random_state)
	
	if use_randomized:
		nn_best, nn_params, nn_score = tune_model_randomized(
			nn, grids['neural_network']['randomized'],
			X_train, y_train, n_iter=n_iter_randomized, cv_fold=cv_fold, random_state=random_state, verbose=0
		)
		if verbose:
			print(f"RandomizedSearchCV best score: {nn_score:.4f}")
			print(f"Best params: {nn_params}")
		
		if use_grid:
			nn_best, nn_params, nn_score = tune_model_grid(
				nn_best, grids['neural_network']['grid'], X_train, y_train, cv_fold=cv_fold, verbose=0
			)
			if verbose:
				print(f"GridSearchCV best score: {nn_score:.4f}")
				print(f"Refined params: {nn_params}")
	else:
		nn_best = nn
		nn_best.fit(X_train, y_train)
	
	models['neural_network'] = nn_best
	results['neural_network'] = evaluate_model(
		nn_best, X_train, y_train, X_test, y_test, "Neural Network", verbose=verbose
	)
	model_time = time.time() - model_start_time
	accuracy = accuracy_score(y_test, nn_best.predict(X_test))
	print(f"Neural Network finished, training time: {model_time:.2f}s, accuracy: {accuracy:.4f}")
	
	total_time = time.time() - total_start_time
	
	best_model_name = None
	best_accuracy = 0
	for model_name, model in models.items():
		model_accuracy = accuracy_score(y_test, model.predict(X_test))
		if model_accuracy > best_accuracy:
			best_accuracy = model_accuracy
			best_model_name = model_name
	
	model_name_display = {
		'logistic_regression': 'Logistic Regression',
		'random_forest': 'Random Forest',
		'gradient_boosting': 'Gradient Boosting',
		'neural_network': 'Neural Network'
	}
	
	print(f"\nTotal training time: {total_time:.2f}s, most accurate: {model_name_display.get(best_model_name, best_model_name)} ({best_accuracy:.4f})")
	
	return models, results

def create_stacked_model(models_dict, X_train, y_train, random_state=42, calibrate=False):
	"""
	Create stacked ensemble model using Random Forest + Gradient Boosting + Logistic Regression.
	
	Args:
		models_dict: Dictionary containing trained models
		X_train: Training features
		y_train: Training target
		random_state: Random seed
		calibrate: If True, calibrate models for reliable probability outputs
	
	Returns:
		Stacked model (VotingClassifier)
	"""
	# Get base models
	rf = models_dict.get('random_forest')
	gb = models_dict.get('gradient_boosting')
	lr = models_dict.get('logistic_regression')
	
	if rf is None or gb is None or lr is None:
		raise ValueError("Need random_forest, gradient_boosting, and logistic_regression models")
	
	if calibrate:
		rf_calibrated = CalibratedClassifierCV(rf, method='isotonic', cv=3)
		gb_calibrated = CalibratedClassifierCV(gb, method='isotonic', cv=3)
		lr_calibrated = CalibratedClassifierCV(lr, method='isotonic', cv=3)
		
		rf_calibrated.fit(X_train, y_train)
		gb_calibrated.fit(X_train, y_train)
		lr_calibrated.fit(X_train, y_train)
		
		estimators = [
			('rf', rf_calibrated),
			('gb', gb_calibrated),
			('lr', lr_calibrated)
		]
	else:
		estimators = [
			('rf', rf),
			('gb', gb),
			('lr', lr)
		]
	
	# Create voting classifier (soft voting for probabilities)
	stacked_model = VotingClassifier(
		estimators=estimators,
		voting='soft',
		weights=None
	)
	
	stacked_model.fit(X_train, y_train)
	return stacked_model

def get_feature_importance(model, feature_names, method='permutation', X_test=None, y_test=None):
	"""
	Get feature importance using different methods.
	
	Args:
		model: Trained model
		feature_names: List of feature names
		method: 'permutation', 'l1', or 'tree' (for tree-based models)
		X_test: Test features (needed for permutation importance)
		y_test: Test target (needed for permutation importance)
	
	Returns:
		DataFrame with feature importances
	"""
	from sklearn.inspection import permutation_importance
	
	if method == 'permutation' and X_test is not None and y_test is not None:
		perm_importance = permutation_importance(
			model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
		)
		importances = perm_importance.importances_mean
		std = perm_importance.importances_std
		
		importance_df = pd.DataFrame({
			'feature': feature_names,
			'importance': importances,
			'std': std
		}).sort_values('importance', ascending=False)
		
	elif method == 'l1' and hasattr(model, 'coef_'):
		# L1 importance for linear models
		coef = np.abs(model.coef_[0])
		importance_df = pd.DataFrame({
			'feature': feature_names,
			'importance': coef
		}).sort_values('importance', ascending=False)
		
	elif method == 'tree' and hasattr(model, 'feature_importances_'):
		# Tree-based feature importance
		importance_df = pd.DataFrame({
			'feature': feature_names,
			'importance': model.feature_importances_
		}).sort_values('importance', ascending=False)
		
	else:
		raise ValueError(f"Method {method} not available for this model type")
	
	return importance_df

def drop_negligible_features(X_train, X_test, feature_importance_df, threshold=0.01):
	"""
	Drop features with importance below threshold to reduce overfitting.
	
	Args:
		X_train: Training features
		X_test: Test features
		feature_importance_df: DataFrame with feature importances
		threshold: Minimum importance threshold (relative to max)
	
	Returns:
		Filtered X_train, X_test, and list of dropped features
	"""
	max_importance = feature_importance_df['importance'].max()
	threshold_value = max_importance * threshold
	
	important_features = feature_importance_df[
		feature_importance_df['importance'] >= threshold_value
	]['feature'].tolist()
	
	dropped_features = [f for f in X_train.columns if f not in important_features]
	
	X_train_filtered = X_train[important_features]
	X_test_filtered = X_test[important_features]
	
	return X_train_filtered, X_test_filtered, dropped_features