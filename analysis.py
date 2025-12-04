import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy import stats
from sklearn.metrics import (
	roc_auc_score, precision_score, recall_score, f1_score,
	confusion_matrix, average_precision_score, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import clone
from models import get_feature_importance

def comprehensive_evaluation(model, X_test, y_test, model_name="Model", 
							cost_false_positive=100, cost_false_negative=5000, verbose=True):
	"""
	Comprehensive evaluation of a model with all relevant metrics.
	
	Args:
		model: Trained model
		X_test: Test features
		y_test: Test target
		model_name: Name of the model for display
		cost_false_positive: Business cost of false positive (incorrectly predicting default)
		cost_false_negative: Business cost of false negative (missing a default)
		verbose: If True, print all metrics
	
	Returns:
		Dictionary containing all evaluation metrics
	"""
	y_pred = model.predict(X_test)
	y_proba = model.predict_proba(X_test)[:, 1]
	
	roc_auc = roc_auc_score(y_test, y_proba)
	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)
	pr_auc = average_precision_score(y_test, y_proba)
	brier_score = brier_score_loss(y_test, y_proba)
	cm = confusion_matrix(y_test, y_pred)
	
	tn, fp, fn, tp = cm.ravel()
	
	total_cost = (fp * cost_false_positive) + (fn * cost_false_negative)
	cost_per_prediction = total_cost / len(y_test)
	
	results = {
		'roc_auc': roc_auc,
		'precision': precision,
		'recall': recall,
		'f1': f1,
		'pr_auc': pr_auc,
		'brier_score': brier_score,
		'confusion_matrix': cm,
		'true_negatives': tn,
		'false_positives': fp,
		'false_negatives': fn,
		'true_positives': tp,
		'total_cost': total_cost,
		'cost_per_prediction': cost_per_prediction
	}
	
	if verbose:
		print(f"\n{model_name} Comprehensive Evaluation")
		
		print(f"\nROC-AUC: {roc_auc:.4f}")
		print("Overall discrimination ability")
		
		print(f"\nPrecision: {precision:.4f}")
		print(f"Recall: {recall:.4f}")
		print(f"F1 Score: {f1:.4f}")
		print("Important when default is minority class or false negatives are costly")
		
		print(f"\nPR-AUC: {pr_auc:.4f}")
		print("Precision-Recall AUC - better than ROC-AUC for class imbalance")
		
		print(f"\nConfusion Matrix:")
		print(f"                Predicted")
		print(f"              No Default  Default")
		print(f"Actual No Default    {tn:5d}     {fp:5d}")
		print(f"       Default       {fn:5d}     {tp:5d}")
		print(f"\nFalse Positives: {fp}")
		print(f"False Negatives: {fn}")
		
		print(f"\nBrier Score: {brier_score:.4f}")
		print("Lower is better - measures probabilistic prediction quality")
		
		print(f"\nCost Analysis:")
		print(f"Cost per False Positive: ${cost_false_positive:,.2f}")
		print(f"Cost per False Negative: ${cost_false_negative:,.2f}")
		print(f"Total False Positives: {fp}")
		print(f"Total False Negatives: {fn}")
		print(f"Total Business Cost: ${total_cost:,.2f}")
		print(f"Cost per Prediction: ${cost_per_prediction:,.2f}")
		print(f"\nPotential writeoff (if all defaults missed): ${fn * cost_false_negative:,.2f}")
		print(f"Potential payback (if all correctly identified): ${tp * cost_false_negative:,.2f}")
	
	return results

def predict_default_probability(model, LIMIT_BAL=None, SEX=None, EDUCATION=None, MARRIAGE=None, 
					   AGE=None, PAY_0=None, PAY_2=None, PAY_3=None, PAY_4=None, 
					   PAY_5=None, PAY_6=None, BILL_AMT1=None, BILL_AMT2=None, 
					   BILL_AMT3=None, BILL_AMT4=None, BILL_AMT5=None, BILL_AMT6=None,
					   PAY_AMT1=None, PAY_AMT2=None, PAY_AMT3=None, PAY_AMT4=None,
					   PAY_AMT5=None, PAY_AMT6=None, feature_names=None, scaler=None):
	"""
	Predict default probability for a hypothetical person.
	
	Args:
		model: Trained model
		LIMIT_BAL: Credit limit balance
		SEX: Sex (1=male, 2=female)
		EDUCATION: Education level (1=graduate, 2=university, 3=high school, 4=others)
		MARRIAGE: Marital status (1=married, 2=single, 3=others)
		AGE: Age in years
		PAY_0 through PAY_6: Payment status (-1=pay duly, 0=no consumption, 1+=months overdue)
		BILL_AMT1 through BILL_AMT6: Bill statement amounts
		PAY_AMT1 through PAY_AMT6: Previous payment amounts
		feature_names: List of feature names in the order expected by the model
		scaler: Scaler object (StandardScaler or RobustScaler) used during training. Required if models were trained on scaled data.
	
	Returns:
		Probability of default (0-1)
	"""
	if feature_names is None:
		raise ValueError("feature_names must be provided - use X_train.columns.tolist() from your training data")
	
	feature_dict = {
		'LIMIT_BAL': LIMIT_BAL,
		'SEX': SEX,
		'EDUCATION': EDUCATION,
		'MARRIAGE': MARRIAGE,
		'AGE': AGE,
		'PAY_0': PAY_0,
		'PAY_2': PAY_2,
		'PAY_3': PAY_3,
		'PAY_4': PAY_4,
		'PAY_5': PAY_5,
		'PAY_6': PAY_6,
		'BILL_AMT1': BILL_AMT1,
		'BILL_AMT2': BILL_AMT2,
		'BILL_AMT3': BILL_AMT3,
		'BILL_AMT4': BILL_AMT4,
		'BILL_AMT5': BILL_AMT5,
		'BILL_AMT6': BILL_AMT6,
		'PAY_AMT1': PAY_AMT1,
		'PAY_AMT2': PAY_AMT2,
		'PAY_AMT3': PAY_AMT3,
		'PAY_AMT4': PAY_AMT4,
		'PAY_AMT5': PAY_AMT5,
		'PAY_AMT6': PAY_AMT6
	}
	
	feature_values = []
	for feature in feature_names:
		if feature in feature_dict:
			value = feature_dict[feature]
			if value is None:
				raise ValueError(f"Missing required feature: {feature}")
			feature_values.append(value)
		else:
			if feature == 'default_payment_next_month':
				continue
			if feature.startswith('SEX_') or feature.startswith('EDUCATION_') or feature.startswith('MARRIAGE_'):
				base_feature = feature.split('_')[0]
				if base_feature == 'SEX' and SEX is not None:
					feature_values.append(1 if feature == f'SEX_{SEX}' else 0)
				elif base_feature == 'EDUCATION' and EDUCATION is not None:
					feature_values.append(1 if feature == f'EDUCATION_{EDUCATION}' else 0)
				elif base_feature == 'MARRIAGE' and MARRIAGE is not None:
					feature_values.append(1 if feature == f'MARRIAGE_{MARRIAGE}' else 0)
				else:
					feature_values.append(0)
			elif 'total_bills_paid_to_date' in feature:
				if PAY_AMT1 is not None:
					feature_values.append(sum([PAY_AMT1 or 0, PAY_AMT2 or 0, PAY_AMT3 or 0, 
											  PAY_AMT4 or 0, PAY_AMT5 or 0, PAY_AMT6 or 0]))
				else:
					feature_values.append(0)
			elif 'avg_bill' in feature:
				if BILL_AMT1 is not None:
					bills = [BILL_AMT1 or 0, BILL_AMT2 or 0, BILL_AMT3 or 0, 
							BILL_AMT4 or 0, BILL_AMT5 or 0, BILL_AMT6 or 0]
					feature_values.append(np.mean(bills))
				else:
					feature_values.append(0)
			elif 'current_outstanding' in feature:
				if BILL_AMT1 is not None:
					feature_values.append(sum([BILL_AMT1 or 0, BILL_AMT2 or 0, BILL_AMT3 or 0, 
											  BILL_AMT4 or 0, BILL_AMT5 or 0, BILL_AMT6 or 0]))
				else:
					feature_values.append(0)
			elif 'amortised_debt' in feature:
				if BILL_AMT1 is not None and PAY_AMT1 is not None:
					bills = sum([BILL_AMT1 or 0, BILL_AMT2 or 0, BILL_AMT3 or 0, 
								BILL_AMT4 or 0, BILL_AMT5 or 0, BILL_AMT6 or 0])
					payments = sum([PAY_AMT1 or 0, PAY_AMT2 or 0, PAY_AMT3 or 0, 
								   PAY_AMT4 or 0, PAY_AMT5 or 0, PAY_AMT6 or 0])
					feature_values.append(max(0, bills - payments))
				else:
					feature_values.append(0)
			elif 'avg_overdue_months' in feature:
				if PAY_0 is not None:
					pay_status = [PAY_0 or 0, PAY_2 or 0, PAY_3 or 0, PAY_4 or 0, PAY_5 or 0, PAY_6 or 0]
					feature_values.append(np.mean(pay_status))
				else:
					feature_values.append(0)
			elif 'max_overdue_months' in feature:
				if PAY_0 is not None:
					pay_status = [PAY_0 or 0, PAY_2 or 0, PAY_3 or 0, PAY_4 or 0, PAY_5 or 0, PAY_6 or 0]
					feature_values.append(max(pay_status))
				else:
					feature_values.append(0)
			elif 'months_overdue_count' in feature:
				if PAY_0 is not None:
					pay_status = [PAY_0 or 0, PAY_2 or 0, PAY_3 or 0, PAY_4 or 0, PAY_5 or 0, PAY_6 or 0]
					feature_values.append(sum(1 for p in pay_status if p > 0))
				else:
					feature_values.append(0)
			elif 'credit_utilisation' in feature:
				if LIMIT_BAL is not None and LIMIT_BAL > 0:
					if BILL_AMT1 is not None:
						bills = sum([BILL_AMT1 or 0, BILL_AMT2 or 0, BILL_AMT3 or 0, 
									BILL_AMT4 or 0, BILL_AMT5 or 0, BILL_AMT6 or 0])
						feature_values.append(min(1.0, bills / LIMIT_BAL))
					else:
						feature_values.append(0)
				else:
					feature_values.append(0)
			else:
				feature_values.append(0)
	
	feature_array = np.array([feature_values])
	
	# Apply scaling if scaler is provided (models were trained on scaled data)
	if scaler is not None:
		feature_array = scaler.transform(feature_array)
	
	probability = model.predict_proba(feature_array)[0, 1]
	
	return probability

def cross_validate_models(models_dict, X_train, y_train, cv=5, random_state=42, verbose=True):
	"""
	Cross-validate all models using the same stratified CV folds for fairness.
	Reports ROC-AUC and PR-AUC with mean ± std.
	
	Args:
		models_dict: Dictionary of trained models
		X_train: Training features
		y_train: Training target
		cv: Number of CV folds
		random_state: Random seed
		verbose: If True, print results
	
	Returns:
		Dictionary with CV results for each model
	"""
	cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
	results = {}
	
	if verbose:
		print("\nCross-Validation Evaluation (Same Folds for All Models)")
	
	for model_name, model in models_dict.items():
		roc_auc_scores = []
		pr_auc_scores = []
		
		model_clone = clone(model)
		
		for fold_idx, (train_idx, val_idx) in enumerate(cv_fold.split(X_train, y_train)):
			if isinstance(X_train, pd.DataFrame):
				X_fold_train = X_train.iloc[train_idx]
				X_fold_val = X_train.iloc[val_idx]
			else:
				X_fold_train = X_train[train_idx]
				X_fold_val = X_train[val_idx]
			
			if isinstance(y_train, pd.Series):
				y_fold_train = y_train.iloc[train_idx]
				y_fold_val = y_train.iloc[val_idx]
			else:
				y_fold_train = y_train[train_idx]
				y_fold_val = y_train[val_idx]
			
			model_clone.fit(X_fold_train, y_fold_train)
			y_proba = model_clone.predict_proba(X_fold_val)[:, 1]
			
			roc_auc = roc_auc_score(y_fold_val, y_proba)
			pr_auc = average_precision_score(y_fold_val, y_proba)
			
			roc_auc_scores.append(roc_auc)
			pr_auc_scores.append(pr_auc)
		
		roc_mean = np.mean(roc_auc_scores)
		roc_std = np.std(roc_auc_scores)
		pr_mean = np.mean(pr_auc_scores)
		pr_std = np.std(pr_auc_scores)
		
		results[model_name] = {
			'roc_auc_scores': roc_auc_scores,
			'pr_auc_scores': pr_auc_scores,
			'roc_auc_mean': roc_mean,
			'roc_auc_std': roc_std,
			'pr_auc_mean': pr_mean,
			'pr_auc_std': pr_std
		}
		
		if verbose:
			display_name = {
				'logistic_regression': 'Logistic Regression',
				'random_forest': 'Random Forest',
				'gradient_boosting': 'Gradient Boosting',
				'neural_network': 'Neural Network'
			}.get(model_name, model_name)
			
			print(f"\n{display_name}:")
			print(f"  ROC-AUC: {roc_mean:.4f} ± {roc_std:.4f}")
			print(f"  PR-AUC:  {pr_mean:.4f} ± {pr_std:.4f}")
	
	return results

def mcnemar_test(y_true, y_pred1, y_pred2):
	"""
	McNemar's test for comparing two classifiers on the same dataset.
	
	Args:
		y_true: True labels
		y_pred1: Predictions from model 1
		y_pred2: Predictions from model 2
	
	Returns:
		chi2 statistic, p-value
	"""
	correct1 = (y_true == y_pred1)
	correct2 = (y_true == y_pred2)
	
	b = np.sum((correct1 == False) & (correct2 == True))
	c = np.sum((correct1 == True) & (correct2 == False))
	
	if b + c == 0:
		return 0.0, 1.0
	
	chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
	p_value = 1 - stats.chi2.cdf(chi2, df=1)
	
	return chi2, p_value

def paired_t_test(scores1, scores2):
	"""
	Paired t-test for comparing two sets of scores (e.g., from CV folds).
	
	Args:
		scores1: Scores from model 1 (array-like)
		scores2: Scores from model 2 (array-like)
	
	Returns:
		t statistic, p-value
	"""
	scores1 = np.array(scores1)
	scores2 = np.array(scores2)
	
	differences = scores1 - scores2
	t_stat, p_value = stats.ttest_1samp(differences, 0)
	
	return t_stat, p_value

def statistical_comparison(models_dict, X_test, y_test, cv_results=None, verbose=True):
	"""
	Perform McNemar's test and paired t-test to compare models.
	
	Args:
		models_dict: Dictionary of trained models
		X_test: Test features
		y_test: Test target
		cv_results: Optional CV results from cross_validate_models
		verbose: If True, print results
	
	Returns:
		Dictionary with statistical test results
	"""
	if verbose:
		print("\nStatistical Model Comparison")
	
	model_names = list(models_dict.keys())
	predictions = {}
	
	for model_name, model in models_dict.items():
		predictions[model_name] = model.predict(X_test)
	
	comparison_results = {}
	
	for i, model1_name in enumerate(model_names):
		for model2_name in model_names[i+1:]:
			pred1 = predictions[model1_name]
			pred2 = predictions[model2_name]
			
			chi2, p_mcnemar = mcnemar_test(y_test, pred1, pred2)
			
			comparison_key = f"{model1_name}_vs_{model2_name}"
			comparison_results[comparison_key] = {
				'mcnemar_chi2': chi2,
				'mcnemar_p': p_mcnemar
			}
			
			if cv_results is not None:
				if model1_name in cv_results and model2_name in cv_results:
					t_stat, p_ttest = paired_t_test(
						cv_results[model1_name]['roc_auc_scores'],
						cv_results[model2_name]['roc_auc_scores']
					)
					comparison_results[comparison_key]['ttest_t'] = t_stat
					comparison_results[comparison_key]['ttest_p'] = p_ttest
			
			if verbose:
				display1 = {
					'logistic_regression': 'Logistic Regression',
					'random_forest': 'Random Forest',
					'gradient_boosting': 'Gradient Boosting',
					'neural_network': 'Neural Network'
				}.get(model1_name, model1_name)
				
				display2 = {
					'logistic_regression': 'Logistic Regression',
					'random_forest': 'Random Forest',
					'gradient_boosting': 'Gradient Boosting',
					'neural_network': 'Neural Network'
				}.get(model2_name, model2_name)
				
				print(f"\n{display1} vs {display2}:")
				print(f"  McNemar's test: chi2={chi2:.4f}, p={p_mcnemar:.4f}")
				if 'ttest_t' in comparison_results[comparison_key]:
					print(f"  Paired t-test: t={comparison_results[comparison_key]['ttest_t']:.4f}, p={comparison_results[comparison_key]['ttest_p']:.4f}")
	
	return comparison_results

def plot_feature_importance_and_shap(model, X_train, X_test, feature_names, model_name="Model", 
							top_n=20, save_plots=False, y_test=None):
	"""
	Plot feature importance and SHAP values for model interpretation.
	
	Args:
		model: Trained model
		X_train: Training features
		X_test: Test features (for SHAP)
		feature_names: List of feature names
		model_name: Name of model for display
		top_n: Number of top features to show
		save_plots: If True, save plots to files
		y_test: Test target (needed for permutation importance)
	"""
	import os
	
	# Create results folder if saving plots
	if save_plots:
		os.makedirs('results', exist_ok=True)
	
	if hasattr(model, 'feature_importances_'):
		method = 'tree'
		importance_df = get_feature_importance(model, feature_names, method=method)
	else:
		if y_test is not None:
			method = 'permutation'
			importance_df = get_feature_importance(model, feature_names, method=method, X_test=X_test, y_test=y_test)
		else:
			print(f"Warning: y_test not provided, cannot compute permutation importance for {model_name}")
			return
	
	top_features = importance_df.head(top_n)
	
	plt.figure(figsize=(10, 8))
	plt.barh(range(len(top_features)), top_features['importance'].values)
	plt.yticks(range(len(top_features)), top_features['feature'].values)
	plt.xlabel('Importance')
	plt.title(f'{model_name} - Top {top_n} Feature Importances')
	plt.gca().invert_yaxis()
	plt.tight_layout()
	
	if save_plots:
		plt.savefig(f'results/{model_name}_feature_importance.png', dpi=150)
		print(f"Saved feature importance plot to results/{model_name}_feature_importance.png")
	else:
		plt.show()
	
	plt.close()
	
	try:
		X_test_sample = X_test[:100] if hasattr(X_test, 'iloc') else X_test[:100]
		X_train_sample = X_train[:100] if hasattr(X_train, 'iloc') else X_train[:100]
		
		if hasattr(model, 'feature_importances_'):
			explainer = shap.TreeExplainer(model)
			shap_values = explainer.shap_values(X_test_sample)
		elif hasattr(model, 'coef_'):
			explainer = shap.LinearExplainer(model, X_train_sample)
			shap_values = explainer.shap_values(X_test_sample)
		else:
			def model_predict(X):
				return model.predict_proba(X)[:, 1]
			explainer = shap.KernelExplainer(model_predict, X_train_sample)
			shap_values = explainer.shap_values(X_test_sample)
		
		# Handle different SHAP value formats
		if isinstance(shap_values, list):
			if len(shap_values) == 2:
				# Binary classification: use positive class (index 1)
				shap_values = shap_values[1]
			else:
				shap_values = shap_values[0]
		
		shap_values = np.array(shap_values)
		
		# Handle 3D arrays (samples, classes, features) - take positive class
		if len(shap_values.shape) == 3:
			# For binary classification, shape is (n_samples, n_classes, n_features)
			# Take the positive class (index 1) or the last class
			if shap_values.shape[1] == 2:
				shap_values = shap_values[:, 1, :]  # Take positive class
			else:
				shap_values = shap_values[:, -1, :]  # Take last class
		
		# Ensure 2D array (samples, features)
		if len(shap_values.shape) == 1:
			shap_values = shap_values.reshape(1, -1)
		
		# Convert X_test_sample to numpy array and ensure shapes match
		if hasattr(X_test_sample, 'values'):
			X_test_sample_array = X_test_sample.values
		elif hasattr(X_test_sample, 'to_numpy'):
			X_test_sample_array = X_test_sample.to_numpy()
		else:
			X_test_sample_array = np.array(X_test_sample)
		
		# Ensure both are 2D
		if len(X_test_sample_array.shape) == 1:
			X_test_sample_array = X_test_sample_array.reshape(-1, 1)
		
		# Ensure both have the same number of samples (should match, but be safe)
		if shap_values.shape[0] != X_test_sample_array.shape[0]:
			min_samples = min(shap_values.shape[0], X_test_sample_array.shape[0])
			shap_values = shap_values[:min_samples, :]
			X_test_sample_array = X_test_sample_array[:min_samples, :]
		
		# Ensure both have the same number of features
		if shap_values.shape[1] != X_test_sample_array.shape[1]:
			min_features = min(shap_values.shape[1], X_test_sample_array.shape[1])
			shap_values = shap_values[:, :min_features]
			X_test_sample_array = X_test_sample_array[:, :min_features]
			feature_names_plot = feature_names[:min_features]
		else:
			feature_names_plot = feature_names
		
		plt.figure(figsize=(10, 8))
		shap.summary_plot(shap_values, X_test_sample_array, feature_names=feature_names_plot, show=False, max_display=top_n)
		plt.title(f'{model_name} - SHAP Summary Plot')
		plt.tight_layout()
		
		if save_plots:
			plt.savefig(f'results/{model_name}_shap_summary.png', dpi=150, bbox_inches='tight')
			print(f"Saved SHAP summary plot to results/{model_name}_shap_summary.png")
		else:
			plt.show()
		
		plt.close()
		
		# SHAP Bar Plot - use already processed shap_values
		try:
			# shap_values is already processed to 2D array (samples, features)
			# Calculate mean absolute SHAP values per feature
			mean_shap = np.abs(shap_values).mean(axis=0)  # Mean across samples
			
			# Ensure feature names match
			if len(mean_shap) != len(feature_names):
				min_len = min(len(mean_shap), len(feature_names))
				mean_shap = mean_shap[:min_len]
				feature_names_use = feature_names[:min_len]
			else:
				feature_names_use = feature_names
			
			# Get top N features
			top_indices = np.argsort(mean_shap)[::-1][:top_n]
			top_mean_shap = mean_shap[top_indices]
			top_feature_names = [feature_names_use[i] for i in top_indices]
			
			# Create bar plot manually
			plt.figure(figsize=(10, 8))
			plt.barh(range(len(top_indices)), top_mean_shap)
			plt.yticks(range(len(top_indices)), top_feature_names)
			plt.xlabel('Mean |SHAP value|')
			plt.title(f'{model_name} - SHAP Bar Plot')
			plt.gca().invert_yaxis()
			plt.tight_layout()
			
			if save_plots:
				plt.savefig(f'results/{model_name}_shap_bar.png', dpi=150, bbox_inches='tight')
				print(f"Saved SHAP bar plot to results/{model_name}_shap_bar.png")
			else:
				plt.show()
			
			plt.close()
		except Exception as bar_error:
			print(f"Warning: Could not generate SHAP bar plot: {bar_error}")
		
	except Exception as e:
		print(f"Error generating SHAP plots: {e}")