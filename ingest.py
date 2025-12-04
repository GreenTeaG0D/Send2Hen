import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
from imblearn.over_sampling import SMOTE

def preprocess_data(file_path, verbose=False, apply_scaling=False, split_data=False, apply_smote=False, test_size=0.2, random_state=42):
	"""
	Preprocess the credit card default dataset.
	
	Args:
		file_path: Path to the Excel file containing the dataset
		verbose: If True, print progress information. If False, no output.
		apply_scaling: If True, apply robust scaling (robust if anomalies present, else standard)
		split_data: If True, return train/test split. If False, return single DataFrame
		apply_smote: If True, apply SMOTE to training data (only works if split_data=True)
		test_size: Proportion of dataset to use for testing (default 0.2)
		random_state: Random seed for reproducibility
	
	Returns:
		If split_data=False: Preprocessed pandas DataFrame
		If split_data=True: (X_train, X_test, y_train, y_test, scaler) tuple
			scaler will be None if apply_scaling=False, otherwise the fitted scaler object
	"""
	# Helper function for conditional printing
	def log(*args, **kwargs):
		if verbose:
			print(*args, **kwargs)
	
	# Load the dataset
	df = pd.read_excel(file_path)
	
	# Rename columns to their descriptive names
	column_mapping = {
		'X1': 'LIMIT_BAL',
		'X2': 'SEX',
		'X3': 'EDUCATION',
		'X4': 'MARRIAGE',
		'X5': 'AGE',
		'X6': 'PAY_0',
		'X7': 'PAY_2',
		'X8': 'PAY_3',
		'X9': 'PAY_4',
		'X10': 'PAY_5',
		'X11': 'PAY_6',
		'X12': 'BILL_AMT1',
		'X13': 'BILL_AMT2',
		'X14': 'BILL_AMT3',
		'X15': 'BILL_AMT4',
		'X16': 'BILL_AMT5',
		'X17': 'BILL_AMT6',
		'X18': 'PAY_AMT1',
		'X19': 'PAY_AMT2',
		'X20': 'PAY_AMT3',
		'X21': 'PAY_AMT4',
		'X22': 'PAY_AMT5',
		'X23': 'PAY_AMT6',
		'Y': 'default_payment_next_month'
	}
	
	# Filter mapping to only include columns that exist, then rename all at once
	# (Learned this trick on leetcode)
	filtered_mapping = {old: new for old, new in column_mapping.items() if old in df.columns}
	if filtered_mapping:
		df = df.rename(columns=filtered_mapping)
		log("Renamed columns:")
		for old_name, new_name in filtered_mapping.items():
			log(f"  {old_name} -> {new_name}")
		log()
	
	"""
	Load and check the dataset
	"""
	
	log("INITIAL DATA EXPLORATION")
	
	log(f"\nDataFrame head:\n{df.head()}")
	
	
	log(f"\nDataFrame tail:\n{df.tail()}")
	
	log(f"\nDataFrame info:\n{df.info()}")
	
	log(f"\nDataFrame describe:\n{df.describe()}")
	
	log(f"\nDataFrame shape: {df.shape[0]} rows, {df.shape[1]} columns")
	log(f"Expected size: ~30,000 rows")
	if df.shape[0] < 2500: # Minimum requirement of 2,500 rows
		log("WARNING: Dataset size below minimum requirement of 2,500 rows")
	elif 25000 <= df.shape[0] <= 35000:
		log("Dataset size looks good")
	else:
		log(f"Dataset size: {df.shape[0]} rows (expected ~30,000)")
	
	log(f"\nDataFrame columns:\n{df.columns.tolist()}")
	log(df.columns.tolist())
	
	log(f"\nDataFrame dtypes:\n{df.dtypes}")
	
	log(f"\nMissing values count:\n{df.isna().sum()}")
	missing_counts = df.isna().sum()
	log(missing_counts[missing_counts > 0])
	
	log(f"\nDuplicate rows:\n{df.duplicated().sum()}")
	
	"""
	Remove unnecessary columns
	"""
	log("REMOVING UNNECESSARY COLUMNS")
	
	# Remove ID/index columns - they're not useful for prediction
	# Common names: ID, id, Unnamed: 0, index (default for excel files)
	cols_to_drop = []
	for col in df.columns:
		col_lower = str(col).lower()
		if col_lower in ['id', 'unnamed: 0', 'index'] or 'id' in col_lower:
			cols_to_drop.append(col)
	
	if cols_to_drop:
		log(f"Dropping columns: {cols_to_drop}")
		df = df.drop(columns=cols_to_drop)
	else:
		log("No obvious index columns found to drop")
	
	"""
	Handle missing values
	"""
	log("HANDLING MISSING VALUES")
	
	missing_counts = df.isna().sum()
	missing_cols = missing_counts[missing_counts > 0]
	
	if len(missing_cols) == 0:
		log("No missing values found")
	else:
		log(f"Columns with missing values: {len(missing_cols)}")
		
		# First pass: drop columns with too many missing values (10% or more)
		cols_to_drop_missing = []
		for col in missing_cols.index:  
			missing_pct = (missing_counts[col] / len(df)) * 100
			if missing_pct >= 10:
				log(f"{col}: {missing_counts[col]} missing ({missing_pct:.2f}%) - Dropping column")
				cols_to_drop_missing.append(col)
		
		if cols_to_drop_missing:
			log(f"\nDropping columns with too many missing values: {cols_to_drop_missing}")
			df = df.drop(columns=cols_to_drop_missing)
			missing_counts = df.isna().sum()
			missing_cols = missing_counts[missing_counts > 0]
		
		# Check if we need KNN imputation (for numeric columns with 10>X>5% missing)
		numeric_cols_for_knn = []
		for col in missing_cols.index:
			if df[col].dtype in ['int64', 'float64']:
				missing_pct = (missing_counts[col] / len(df)) * 100
				if missing_pct <= 10 and missing_pct > 5:
					numeric_cols_for_knn.append(col)
		
		# Handle KNN imputation for all numeric columns at once if needed
		if numeric_cols_for_knn:
			log(f"\nUsing K-NN imputation for numeric columns with 10>X>5% missing: {numeric_cols_for_knn}")
			imputer = KNNImputer(n_neighbors=5)
			numeric_cols_all = df.select_dtypes(include=[np.number]).columns
			df_numeric = df[numeric_cols_all].copy()
			imputed = imputer.fit_transform(df_numeric)
			df[numeric_cols_all] = imputed
			# Update missing counts after KNN
			missing_counts = df.isna().sum()
			missing_cols = missing_counts[missing_counts > 0]
				
		# Handle remaining missing values column by column
		for col in missing_cols.index:
			missing_pct = (missing_counts[col] / len(df)) * 100
			log(f"\n{col}: {missing_counts[col]} missing ({missing_pct:.2f}%)")
			
			# If insignificant count, just drop the rows
			if missing_pct < 1:
				log(f"  -> Dropping {missing_counts[col]} rows with missing values")
				df = df.dropna(subset=[col])
			else:
				# Handle based on data type
				if df[col].dtype in ['int64', 'float64']:
					# Numeric: use mean (KNN already handled if needed)
					mean_val = df[col].mean()
					log(f"  -> Filling {missing_counts[col]} missing values with mean: {mean_val:.2f}")
					df[col] = df[col].fillna(mean_val)
				else:
					# Categorical: impute with mode
					mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
					log(f"  -> Filling {missing_counts[col]} missing values with mode: {mode_val}")
					df[col] = df[col].fillna(mode_val)
	
	log(f"\nShape after handling missing values: {df.shape}")
	
	"""
	Correct types and encoding
	"""
	log("CORRECTING TYPES AND ENCODING")
	
	# Float -> Int (where values are whole numbers)
	for col in df.select_dtypes(include=['float64']).columns:
		if df[col].notna().all():
			if (df[col] % 1 == 0).all():
				log(f"Converting {col} from float to int")
				df[col] = df[col].astype('int64')
	
	# String -> Int (where applicable - e.g., "1", "2" -> 1, 2)
	string_to_int_count = 0
	for col in df.select_dtypes(include=['object']).columns:
		try:
			# Try converting to numeric
			converted = pd.to_numeric(df[col], errors='coerce')
			if converted.notna().sum() / len(df) > 0.9:  # If 90%+ can be converted
				converted_count = converted.notna().sum()
				log(f"Converting {col} from string to int ({converted_count} values converted)")
				df[col] = converted.fillna(0).astype('int64')
				string_to_int_count += 1
		except:
			pass
	
	if string_to_int_count > 0:
		log(f"Total columns converted from string to int: {string_to_int_count}")
	
	# Identify categorical columns for one-hot encoding
	# Look for columns with limited unique values (nominal categories)
	categorical_cols = []
	for col in df.columns:
		if df[col].dtype == 'object' or df[col].dtype.name == 'category':
			categorical_cols.append(col)
		elif df[col].dtype in ['int64', 'int32']:
			# Could be categorical if low cardinality
			unique_count = df[col].nunique()
			if unique_count <= 10 and unique_count < len(df) * 0.1:
				# Low cardinality, might be categorical
				# But we'll handle payment status separately
				pass
	
	# One-hot encode nominal categories (where order doesn't matter)
	# Common ones: SEX, EDUCATION, MARRIAGE
	nominal_cols = []
	for col in categorical_cols:
		col_lower = str(col).lower()
		if any(x in col_lower for x in ['sex', 'gender', 'education', 'marriage', 'marital']):
			nominal_cols.append(col)
	
	# Also check for other low-cardinality int columns that might be nominal
	for col in df.select_dtypes(include=['int64', 'int32']).columns:
		col_lower = str(col).lower()
		if col_lower in ['sex', 'education', 'marriage'] and df[col].nunique() <= 10:
			nominal_cols.append(col)
	
	log(f"\nOne-hot encoding nominal categories: {nominal_cols}")
	for col in nominal_cols:
		if col in df.columns:
			dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
			df = pd.concat([df, dummies], axis=1)
			df = df.drop(columns=[col])
			log(f"  -> Encoded {col} into {len(dummies.columns)} binary columns")
	
	# Ordinal encode payment status (where order matters)
	# Look for PAY_ columns or payment status columns
	payment_cols = []
	for col in df.columns:
		col_lower = str(col).lower()
		if 'pay' in col_lower and ('status' in col_lower or col_lower.startswith('pay_')):
			payment_cols.append(col)
	
	log(f"\nPayment status columns (keeping as ordinal): {payment_cols}")
	# Payment status is already numeric typically, but we'll ensure it's int
	for col in payment_cols:
		if col in df.columns:
			df[col] = df[col].astype('int64')
			log(f"  -> {col} kept as ordinal (int)")
	
	# Int -> Category (where applicable - for memory efficiency)
	# Only for very low cardinality columns that aren't being used in calculations
	for col in df.select_dtypes(include=['int64', 'int32']).columns:
		if col not in payment_cols and df[col].nunique() <= 5:
			col_lower = str(col).lower()
			# Don't convert if it's a bill amount, payment amount, or limit
			if not any(x in col_lower for x in ['bill', 'pay_amt', 'limit', 'bal']):
				log(f"Converting {col} to category for memory efficiency")
				df[col] = df[col].astype('category')
	
	# Date(ish) -> Date-time
	# Look for date-like columns
	for col in df.select_dtypes(include=['object']).columns:
		try:
			# Try parsing as date
			parsed = pd.to_datetime(df[col], errors='coerce')
			if parsed.notna().sum() / len(df) > 0.5:  # If 50%+ can be parsed
				log(f"Converting {col} to datetime")
				df[col] = parsed
		except:
			pass
	
	log(f"\nShape after type corrections: {df.shape}")
	
	"""
	Anomaly detection
	"""
	
	log("REMOVING OUTLIERS")
	
	initial_rows = len(df)
	
	# Method 1: 2 Standard Deviations (2-SD)
	log("Using 2-SD method:")
	numeric_cols = df.select_dtypes(include=[np.number]).columns
	outliers_2sd = set()
	
	for col in numeric_cols:
		z_scores = np.abs(zscore(df[col].dropna()))
		outlier_mask = z_scores > 2
		outlier_count = outlier_mask.sum()
		if outlier_count > 0:
			outlier_indices = df[col].index[outlier_mask]
			outliers_2sd.update(outlier_indices)
			log(f"  {col}: {outlier_count} outliers (|z| > 2)")
	
	# Method 2: IQR (Interquartile Range)
	log("Using IQR method:")
	outliers_iqr = set()
	
	for col in numeric_cols:
		Q1 = df[col].quantile(0.25)
		Q3 = df[col].quantile(0.75)
		IQR = Q3 - Q1
		lower_bound = Q1 - 1.5 * IQR
		upper_bound = Q3 + 1.5 * IQR
		outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
		outlier_count = outlier_mask.sum()
		if outlier_count > 0:
			outlier_indices = df[col].index[outlier_mask]
			outliers_iqr.update(outlier_indices)
			log(f"  {col}: {outlier_count} outliers (outside IQR bounds)")
	
	# Combine both methods - remove if flagged by either
	all_outliers = outliers_2sd.union(outliers_iqr)
	log(f"\nTotal unique outlier rows: {len(all_outliers)}")
	
	# For now, we'll use winsorization instead of removal to preserve data
	# Winsorize extreme values to 1st and 99th percentiles
	log("\nWinsorizing extreme values (1st and 99th percentiles):")
	total_winsorized = 0
	for col in numeric_cols:
		# Skip if it's a binary/categorical column
		if df[col].nunique() <= 2:
			continue
		
		original_values = df[col].copy()
		original_min = df[col].min()
		original_max = df[col].max()
		
		# Winsorize at 1st and 99th percentiles
		winsorized = winsorize(df[col].values, limits=[0.01, 0.01])
		# Convert masked array to regular array
		df[col] = np.array(winsorized)
		
		# Count how many values were actually changed
		values_changed = (original_values != df[col]).sum()
		if values_changed > 0:
			total_winsorized += values_changed
			log(f"  {col}: {values_changed} values clipped (min: {original_min:.2f}->{df[col].min():.2f}, max: {original_max:.2f}->{df[col].max():.2f})")
	
	log(f"\nTotal values winsorized across all columns: {total_winsorized}")
	
	# Group rare values into "Other" for categorical columns
	log("\nGrouping rare categorical values into 'Other':")
	for col in df.select_dtypes(include=['object', 'category']).columns:
		value_counts = df[col].value_counts()
		# If a value appears in less than 1% of rows, group it as "Other"
		rare_threshold = len(df) * 0.01
		rare_values = value_counts[value_counts < rare_threshold].index.tolist()
		
		if len(rare_values) > 0 and len(rare_values) < len(value_counts):
			log(f"  {col}: grouping {len(rare_values)} rare values into 'Other'")
			df[col] = df[col].replace(rare_values, 'Other')
	
	log(f"\nShape after anomaly handling: {df.shape} (removed {initial_rows - len(df)} rows)")
	
	"""
	Derived Features
	"""
	log("CREATING DERIVED FEATURES")
	
	# Find bill amount columns (BILL_AMT1, BILL_AMT2, etc.) - bill_amt is the default column name for bill amounts
	bill_cols = [col for col in df.columns if 'bill_amt' in str(col).lower()]
	# Also check for PAY_AMT columns (payment amounts) - pay_amt is the default column name for payment amounts
	pay_amt_cols = [col for col in df.columns if 'pay_amt' in str(col).lower()]
	
	log(f"Found bill columns: {bill_cols}")
	log(f"Found payment amount columns: {pay_amt_cols}")
	
	# Total bills paid to date (sum of all payment amounts)
	if pay_amt_cols:
		df['total_bills_paid_to_date'] = df[pay_amt_cols].sum(axis=1)
		log(f"Created: total_bills_paid_to_date")
	
	# Average bill amount
	if bill_cols:
		df['avg_bill'] = df[bill_cols].mean(axis=1)
		log(f"Created: avg_bill")
	
	# Total in processing (current outstanding bills)
	if bill_cols:
		df['current_outstanding'] = df[bill_cols].sum(axis=1)
		log(f"Created: current_outstanding")
	
	# Amortised debt (rough estimate: bills - payments over time)
	if bill_cols and pay_amt_cols:
		# Sum of bills minus sum of payments
		df['amortised_debt'] = df[bill_cols].sum(axis=1) - df[pay_amt_cols].sum(axis=1)
		# Can't have negative amortised debt in this context
		df['amortised_debt'] = df['amortised_debt'].clip(lower=0)
		log(f"Created: amortised_debt")
	
	# Temporal features - payment timing
	# Find payment status columns (PAY_0, PAY_2, etc. or PAY_1, PAY_2, etc.)
	pay_status_cols = [col for col in df.columns if 'pay' in str(col).lower() and col not in pay_amt_cols and 'status' not in str(col).lower()]
	# Filter to actual payment status columns (usually PAY_0, PAY_2, PAY_3, etc.)
	pay_status_cols = [col for col in pay_status_cols if any(x in col for x in ['PAY_', 'Pay_', 'pay_'])]
	
	if pay_status_cols:
		# How overdue on average (negative values = paid early, positive = overdue)
		# Payment status typically: -1 = paid duly, 0 = no consumption, 1+ = months overdue
		df['avg_overdue_months'] = df[pay_status_cols].mean(axis=1)
		log(f"Create: avg_overdue_months")
		
		# Max overdue months
		df['max_overdue_months'] = df[pay_status_cols].max(axis=1)
		log(f"Created: max_overdue_months")
		
		# Count of months with overdue payments
		df['months_overdue_count'] = (df[pay_status_cols] > 0).sum(axis=1)
		log(f"Created: months_overdue_count")
	
	# Credit utilisation
	# Look for limit and balance columns
	limit_col = None
	balance_col = None
	
	for col in df.columns:
		col_lower = str(col).lower()
		if 'limit' in col_lower and 'bal' in col_lower:
			limit_col = col
		if ('balance' in col_lower or 'bal' in col_lower) and 'limit' not in col_lower:
			balance_col = col
	
	if limit_col and balance_col:
		# Avoid division by zero
		df['credit_utilisation'] = np.where(
			df[limit_col] > 0,
			df[balance_col] / df[limit_col],
			0
		)
		# Cap at 1.0 (100% utilisation)
		df['credit_utilisation'] = df['credit_utilisation'].clip(upper=1.0)
		log(f"Created: credit_utilisation (using {limit_col} and {balance_col})")
	elif limit_col:
		# If we have limit but no current balance, use total bills as proxy
		if bill_cols:
			df['credit_utilisation'] = np.where(
				df[limit_col] > 0,
				df[bill_cols].sum(axis=1) / df[limit_col],
				0
			)
			df['credit_utilisation'] = df['credit_utilisation'].clip(upper=1.0)
			log(f"Created: credit_utilisation (using {limit_col} and bill amounts as proxy)")
	
	log(f"\nShaape after creating derived features: {df.shape}")
	
	"""
	Final summary
	"""
	log("FINAL DATA SUMMARY")
	
	log(f"\nFinal shape: {df.shape}")
	log(f"Final columns: {len(df.columns)}")
	log(f"\nColumn names:")
	for i, col in enumerate(df.columns, 1):
		log(f"  {i}. {col}")
	
	log(f"\nFinal dtypes:")
	log(df.dtypes.value_counts())
	
	log(f"\nFinal missing values:")
	final_missing = df.isna().sum()
	if final_missing.sum() > 0:
		log(final_missing[final_missing > 0])
	else:
		log("None")
	
	# Ensure cleaned dataset is still large enough
	log(f"\nDataset size validation:")
	log(f"Minimum requirement: 2,500 rows - Current: {df.shape[0]} rows")
	if df.shape[0] >= 2500:
		log("Dataset size meets minimum requirement")
	else:
		log("WARNING: Dataset size below minimum requirement")
		raise ValueError(f"Dataset too small: {df.shape[0]} rows (minimum: 2,500)")
	
	# Check class balance if target column exists
	target_col = 'default_payment_next_month'
	if target_col in df.columns:
		class_counts = df[target_col].value_counts()
		class_balance = class_counts.min() / class_counts.max()
		log(f"\nClass balance check:")
		log(f"  Class distribution:\n{class_counts}")
		log(f"  Balance ratio: {class_balance:.3f}")
		if class_balance < 0.25:  # Less than 25% means >75% in one class
			log(f"  WARNING: Class imbalance detected (>75% in majority class)")
			log(f"  Consider using class_weight='balanced' or SMOTE")
		else:
			log(f"  Class balance is acceptable")
	
	# Separate features and target if splitting
	if split_data:
		if target_col not in df.columns:
			# For some reason this literally only ever happend with 'default_payment_next_month' and it was better to bodge than fix
			raise ValueError("Cannot split data: target column 'default_payment_next_month' not found")
		
		scaler = None  # Initialize scaler variable
		X = df.drop(columns=[target_col])
		y = df[target_col]
		
		log("\nTRAIN/TEST SPLIT")
		
		# Stratified train/test split
		X_train, X_test, y_train, y_test = train_test_split(
			X, y, 
			test_size=test_size, 
			random_state=random_state,
			stratify=y  # Stratified split to maintain class distribution
		)
		
		log(f"Training set: {X_train.shape[0]} rows, {X_train.shape[1]} features")
		log(f"Test set: {X_test.shape[0]} rows, {X_test.shape[1]} features")
		log(f"Training class distribution:\n{y_train.value_counts()}")
		log(f"Test class distribution:\n{y_test.value_counts()}")
		
		# Apply SMOTE to training data if requested
		if apply_smote:
			log(f"\nApplying SMOTE to training data...")
			try:
				smote = SMOTE(random_state=random_state)
				X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
				log(f"Before SMOTE: {X_train.shape[0]} samples")
				log(f"After SMOTE: {X_train_resampled.shape[0]} samples")
				log(f"Resampled class distribution:\n{pd.Series(y_train_resampled).value_counts()}")
				X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns, index=X_train.index[:len(X_train_resampled)])
				y_train = pd.Series(y_train_resampled, index=X_train.index)
			except Exception as e:
				log(f"WARNING: SMOTE failed: {e}")
				log("Continuing without SMOTE...")
		
		# Apply scaling if requested
		if apply_scaling:
			log("\nAPPLYING SCALING")
			
			# Use robust scaling if anomalies were detected (winsorization was applied)
			# Otherwise use standard scaling
			use_robust = total_winsorized > 0
			
			if use_robust:
				log("Using RobustScaler (anomalies detected in data)")
				scaler = RobustScaler()
			else:
				log("Using StandardScaler")
				scaler = StandardScaler()
			
			# Fit on training data, transform both train and test
			X_train_scaled = scaler.fit_transform(X_train)
			X_test_scaled = scaler.transform(X_test)
			
			# Convert back to DataFrame with original column names
			X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
			X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
			
			log("Scaling applied to training and test sets")
		
		log("\nFINAL SUMMARY")
		log(f"Training set shape: {X_train.shape}")
		log(f"Test set shape: {X_test.shape}")
		log(f"\nDataset ready for modeling!")
		
		# Always return scaler when splitting (None if scaling wasn't applied)
		return X_train, X_test, y_train, y_test, scaler
	
	# Apply scaling to entire dataset if requested (and not splitting)
	if apply_scaling:
		log("\nAPPLYING SCALING")
		
		# Use robust scaling if anomalies were detected (winsorization was applied)
		use_robust = total_winsorized > 0
		
		if use_robust:
			log("Using RobustScaler (anomalies detected in data)")
			scaler = RobustScaler()
		else:
			log("Using StandardScaler")
			scaler = StandardScaler()
		
		# Separate target if it exists
		if target_col in df.columns:
			X = df.drop(columns=[target_col])
			y = df[target_col]
			
			# Scale only features
			X_scaled = scaler.fit_transform(X)
			X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
			
			# Recombine
			df = pd.concat([X_scaled, y], axis=1)
		else:
			# Scale all numeric columns
			numeric_cols = df.select_dtypes(include=[np.number]).columns
			df_scaled = scaler.fit_transform(df[numeric_cols])
			df[numeric_cols] = df_scaled
		
		log("Scaling applied to dataset")
	
	log("\nFINAL SUMMARY")
	log(f"Final shape: {df.shape}")
	log(f"\nDataset ready for modeling!")
	
	return df