#!/usr/bin/env python3
"""
Comprehensive Fairness and Bias Analysis for Loan Datasets
Using Microsoft Fairlearn to assess and mitigate bias in loan approval and repayment models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Fairlearn imports
try:
    from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, false_negative_rate
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.preprocessing import CorrelationRemover
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    from fairlearn.datasets import fetch_adult
    print("✓ Fairlearn imported successfully")
except ImportError:
    print("❌ Fairlearn not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "fairlearn"])
    from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, false_negative_rate
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.preprocessing import CorrelationRemover
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    print("✓ Fairlearn installed and imported")

def load_and_prepare_datasets():
    """Load and prepare both loan datasets for fairness analysis"""
    print("=== LOADING AND PREPARING DATASETS FOR FAIRNESS ANALYSIS ===\n")
    
    # Load datasets from CSV_dataset directory
    try:
        approval_df = pd.read_csv('CSV_dataset/Approval.csv')
        payment_df = pd.read_csv('CSV_dataset/Repayment.csv')
        print(f"✓ Loaded approval dataset: {approval_df.shape}")
        print(f"✓ Loaded payment dataset: {payment_df.shape}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None
    
    # Prepare approval dataset
    print("\n--- Preparing Approval Dataset ---")
    approval_clean = approval_df.copy()
    
    # Remove RiskScore if present
    riskscore_cols = [col for col in approval_clean.columns if 'risk' in col.lower() and 'score' in col.lower()]
    if riskscore_cols:
        approval_clean = approval_clean.drop(columns=riskscore_cols)
        print(f"Removed RiskScore columns: {riskscore_cols}")
    
    # Identify sensitive features for approval dataset
    sensitive_features_approval = []
    if 'Age' in approval_clean.columns:
        # Create age groups
        approval_clean['Age_Group'] = pd.cut(approval_clean['Age'], 
                                           bins=[0, 25, 35, 45, 55, 100], 
                                           labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        sensitive_features_approval.append('Age_Group')
    
    if 'MaritalStatus' in approval_clean.columns:
        sensitive_features_approval.append('MaritalStatus')
    
    if 'EducationLevel' in approval_clean.columns:
        sensitive_features_approval.append('EducationLevel')
    
    if 'EmploymentStatus' in approval_clean.columns:
        sensitive_features_approval.append('EmploymentStatus')
    
    print(f"Sensitive features for approval: {sensitive_features_approval}")
    
    # Prepare payment dataset
    print("\n--- Preparing Payment Dataset ---")
    payment_clean = payment_df.copy()
    
    # Remove RiskScore if present
    riskscore_cols = [col for col in payment_clean.columns if 'risk' in col.lower() and 'score' in col.lower()]
    if riskscore_cols:
        payment_clean = payment_clean.drop(columns=riskscore_cols)
        print(f"Removed RiskScore columns: {riskscore_cols}")
    
    # Identify sensitive features for payment dataset
    sensitive_features_payment = []
    if 'Age' in payment_clean.columns:
        # Create age groups
        payment_clean['Age_Group'] = pd.cut(payment_clean['Age'], 
                                          bins=[0, 25, 35, 45, 55, 100], 
                                          labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        sensitive_features_payment.append('Age_Group')
    
    if 'MaritalStatus' in payment_clean.columns:
        sensitive_features_payment.append('MaritalStatus')
    
    if 'EducationLevel' in payment_clean.columns:
        sensitive_features_payment.append('EducationLevel')
    
    if 'EmploymentStatus' in payment_clean.columns:
        sensitive_features_payment.append('EmploymentStatus')
    
    print(f"Sensitive features for payment: {sensitive_features_payment}")
    
    return (approval_clean, sensitive_features_approval), (payment_clean, sensitive_features_payment)

def perform_cross_validation_analysis(X, y, model, dataset_name):
    """Perform cross-validation analysis to check for sampling bias"""
    print(f"\n=== CROSS-VALIDATION ANALYSIS: {dataset_name} ===")
    
    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='precision')
    
    print(f"5-Fold CV Precision Scores: {cv_scores}")
    print(f"Mean CV Precision: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Check for high variance (potential sampling bias)
    cv_std = cv_scores.std()
    if cv_std > 0.05:
        print(f"⚠️  HIGH VARIANCE DETECTED: CV std = {cv_std:.4f}")
        print("   This suggests potential sampling bias or data instability")
    else:
        print(f"✓ Low variance: CV std = {cv_std:.4f}")
    
    return cv_scores

def create_learning_curves(X, y, model, dataset_name):
    """Create learning curves to analyze performance under varying training sizes"""
    print(f"\n=== LEARNING CURVE ANALYSIS: {dataset_name} ===")
    
    # Create learning curve
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=5, 
        scoring='precision', n_jobs=-1
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    plt.xlabel('Training Set Size')
    plt.ylabel('Precision Score')
    plt.title(f'Learning Curve - {dataset_name}')
    plt.legend()
    plt.grid(True)
    
    # Check for overfitting/underfitting
    final_train_score = train_mean[-1]
    final_val_score = val_mean[-1]
    gap = final_train_score - final_val_score
    
    print(f"Final Training Score: {final_train_score:.4f}")
    print(f"Final Validation Score: {final_val_score:.4f}")
    print(f"Gap: {gap:.4f}")
    
    if gap > 0.1:
        print("⚠️  POTENTIAL OVERFITTING: Large gap between training and validation scores")
    elif final_val_score < 0.7:
        print("⚠️  POTENTIAL UNDERFITTING: Low validation score suggests model needs more complexity")
    else:
        print("✓ Model shows good generalization")
    
    return train_sizes_abs, train_mean, val_mean

def perform_fairness_assessment(X, y, sensitive_features, dataset_name):
    """Perform comprehensive fairness assessment using Fairlearn"""
    print(f"\n=== FAIRNESS ASSESSMENT: {dataset_name} ===")
    
    if not sensitive_features:
        print("No sensitive features identified for fairness assessment")
        return None
    
    # Prepare data for fairness assessment
    # Use the first sensitive feature for initial analysis
    sensitive_feature = sensitive_features[0]
    
    # Check if sensitive feature exists and has reasonable distribution
    if sensitive_feature not in X.columns:
        print(f"Sensitive feature {sensitive_feature} not found in dataset")
        return None
    
    # Get unique values and check distribution
    unique_values = X[sensitive_feature].value_counts()
    print(f"Sensitive feature '{sensitive_feature}' distribution:")
    print(unique_values)
    
    # Filter out groups with very few samples
    min_group_size = 50
    valid_groups = unique_values[unique_values >= min_group_size].index
    if len(valid_groups) < 2:
        print(f"⚠️  Not enough samples in groups for fairness assessment")
        return None
    
    # Filter data to include only valid groups
    mask = X[sensitive_feature].isin(valid_groups)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    print(f"Filtered dataset: {len(X_filtered)} samples, {len(valid_groups)} groups")
    
    # Train a simple model for fairness assessment
    # Prepare features (exclude sensitive features for training)
    X_features = X_filtered.drop(columns=sensitive_features)
    
    # Handle categorical variables
    for col in X_features.select_dtypes(include=['object']).columns:
        X_features = pd.get_dummies(X_features, columns=[col], drop_first=True)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_features_imputed = pd.DataFrame(imputer.fit_transform(X_features), columns=X_features.columns)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_features_imputed, y_filtered)
    
    # Get predictions
    y_pred = model.predict(X_features_imputed)
    
    # Create MetricFrame for fairness assessment
    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'selection_rate': selection_rate,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate
    }
    
    # Calculate metrics by group
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_filtered,
        y_pred=y_pred,
        sensitive_features=X_filtered[sensitive_feature]
    )
    
    print(f"\n--- Fairness Metrics by {sensitive_feature} ---")
    print(metric_frame.by_group)
    
    # Calculate disparities
    print(f"\n--- Disparity Analysis ---")
    disparities = metric_frame.difference()
    print(disparities)
    
    # Identify significant disparities
    significant_disparities = []
    for metric in ['precision', 'recall', 'f1', 'selection_rate']:
        if metric in disparities.columns:
            max_diff = disparities[metric].max()
            if max_diff > 0.1:  # 10% threshold
                significant_disparities.append((metric, max_diff))
                print(f"⚠️  SIGNIFICANT DISPARITY in {metric}: {max_diff:.4f}")
    
    if not significant_disparities:
        print("✓ No significant disparities detected")
    
    # Create visualization
    create_fairness_visualization(metric_frame, sensitive_feature, dataset_name)
    
    return metric_frame, disparities

def create_fairness_visualization(metric_frame, sensitive_feature, dataset_name):
    """Create visualizations for fairness assessment"""
    print(f"\n--- Creating Fairness Visualizations ---")
    
    # Plot metrics by group
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Fairness Assessment: {dataset_name} by {sensitive_feature}', fontsize=16)
    
    # Plot 1: Precision by group
    metric_frame.by_group['precision'].plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Precision by Group')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Recall by group
    metric_frame.by_group['recall'].plot(kind='bar', ax=axes[0, 1], color='lightcoral')
    axes[0, 1].set_title('Recall by Group')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Selection Rate by group
    metric_frame.by_group['selection_rate'].plot(kind='bar', ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('Selection Rate by Group')
    axes[1, 0].set_ylabel('Selection Rate')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: F1 Score by group
    metric_frame.by_group['f1'].plot(kind='bar', ax=axes[1, 1], color='gold')
    axes[1, 1].set_title('F1 Score by Group')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'fairness_assessment_{dataset_name.replace(" ", "_").lower()}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Fairness visualization saved as: {plot_filename}")
    
    plt.show()

def analyze_data_balancing_bias(original_df, balanced_df, target_col, dataset_name):
    """Analyze potential bias introduced by data balancing"""
    print(f"\n=== DATA BALANCING BIAS ANALYSIS: {dataset_name} ===")
    
    # Compare original vs balanced distributions
    original_dist = original_df[target_col].value_counts(normalize=True)
    balanced_dist = balanced_df[target_col].value_counts(normalize=True)
    
    print(f"Original distribution: {original_dist.to_dict()}")
    print(f"Balanced distribution: {balanced_dist.to_dict()}")
    
    # Check if sensitive features distribution changed
    sensitive_cols = ['Age', 'MaritalStatus', 'EducationLevel', 'EmploymentStatus']
    available_sensitive = [col for col in sensitive_cols if col in original_df.columns]
    
    if available_sensitive:
        print(f"\n--- Sensitive Feature Distribution Changes ---")
        for col in available_sensitive:
            if col in original_df.columns and col in balanced_df.columns:
                orig_dist = original_df[col].value_counts(normalize=True)
                bal_dist = balanced_df[col].value_counts(normalize=True)
                
                # Calculate KL divergence to measure distribution change
                from scipy.stats import entropy
                kl_div = entropy(bal_dist, orig_dist)
                
                print(f"{col}:")
                print(f"  Original: {orig_dist.head(3).to_dict()}")
                print(f"  Balanced: {bal_dist.head(3).to_dict()}")
                print(f"  KL Divergence: {kl_div:.4f}")
                
                if kl_div > 0.1:
                    print(f"  ⚠️  SIGNIFICANT DISTRIBUTION CHANGE detected")
                else:
                    print(f"  ✓ Distribution relatively preserved")
    
    # Check for potential overfitting due to duplication
    minority_class = original_dist.idxmin()
    minority_original = original_df[original_df[target_col] == minority_class]
    minority_balanced = balanced_df[balanced_df[target_col] == minority_class]
    
    # Check for exact duplicates
    if len(minority_balanced) > len(minority_original):
        # This is expected due to upsampling
        duplication_ratio = len(minority_balanced) / len(minority_original)
        print(f"\nDuplication ratio for minority class: {duplication_ratio:.2f}x")
        
        if duplication_ratio > 3:
            print("⚠️  HIGH DUPLICATION RATIO - may lead to overfitting")
        else:
            print("✓ Reasonable duplication ratio")

def main():
    """Main function to perform comprehensive fairness and bias analysis"""
    print("🔍 COMPREHENSIVE FAIRNESS AND BIAS ANALYSIS")
    print("=" * 60)
    
    # Load and prepare datasets
    (approval_df, approval_sensitive), (payment_df, payment_sensitive) = load_and_prepare_datasets()
    
    if approval_df is None or payment_df is None:
        print("❌ Failed to load datasets")
        return
    
    # Analyze both datasets
    datasets = [
        (approval_df, approval_sensitive, 'LoanApproved', 'Approval'),
        (payment_df, payment_sensitive, 'fully.paid', 'Payment')
    ]
    
    results = {}
    
    for df, sensitive_features, target_col, dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"ANALYZING {dataset_name.upper()} DATASET")
        print(f"{'='*60}")
        
        # Prepare features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical variables
        X_processed = X.copy()
        for col in X_processed.select_dtypes(include=['object']).columns:
            if col not in sensitive_features:  # Don't encode sensitive features yet
                X_processed = pd.get_dummies(X_processed, columns=[col], drop_first=True)
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X_processed), columns=X_processed.columns)
        
        # 1. Cross-validation analysis
        model = LogisticRegression(random_state=42, max_iter=1000)
        cv_scores = perform_cross_validation_analysis(X_imputed, y, model, f"{dataset_name} Dataset")
        
        # 2. Learning curve analysis
        train_sizes, train_scores, val_scores = create_learning_curves(X_imputed, y, model, f"{dataset_name} Dataset")
        
        # 3. Fairness assessment
        fairness_results = perform_fairness_assessment(X, y, sensitive_features, f"{dataset_name} Dataset")
        
        # 4. Data balancing bias analysis (if balanced datasets exist)
        try:
            balanced_filename = f'Loan_{dataset_name.upper()}_ohe_balanced_50_50.csv'
            balanced_df = pd.read_csv(balanced_filename)
            analyze_data_balancing_bias(df, balanced_df, target_col, f"{dataset_name} Dataset")
        except FileNotFoundError:
            print(f"Balanced dataset {balanced_filename} not found - skipping balancing bias analysis")
        
        # Store results
        results[dataset_name] = {
            'cv_scores': cv_scores,
            'learning_curve': (train_sizes, train_scores, val_scores),
            'fairness': fairness_results
        }
    
    # Generate comprehensive report
    generate_bias_analysis_report(results)
    
    print("\n✅ Comprehensive fairness and bias analysis completed!")
    print("📁 Check the generated visualizations and reports for detailed findings.")

def generate_bias_analysis_report(results):
    """Generate a comprehensive bias analysis report"""
    print(f"\n{'='*60}")
    print("COMPREHENSIVE BIAS ANALYSIS REPORT")
    print(f"{'='*60}")
    
    for dataset_name, data in results.items():
        print(f"\n--- {dataset_name.upper()} DATASET FINDINGS ---")
        
        # Cross-validation findings
        cv_scores = data['cv_scores']
        cv_std = cv_scores.std()
        print(f"Cross-Validation Analysis:")
        print(f"  Mean Precision: {cv_scores.mean():.4f}")
        print(f"  Standard Deviation: {cv_std:.4f}")
        if cv_std > 0.05:
            print(f"  ⚠️  HIGH VARIANCE - Potential sampling bias detected")
        else:
            print(f"  ✓ Low variance - Good sampling stability")
        
        # Learning curve findings
        train_sizes, train_scores, val_scores = data['learning_curve']
        final_gap = train_scores[-1] - val_scores[-1]
        print(f"Learning Curve Analysis:")
        print(f"  Final Training Score: {train_scores[-1]:.4f}")
        print(f"  Final Validation Score: {val_scores[-1]:.4f}")
        print(f"  Generalization Gap: {final_gap:.4f}")
        if final_gap > 0.1:
            print(f"  ⚠️  OVERFITTING detected")
        elif val_scores[-1] < 0.7:
            print(f"  ⚠️  UNDERFITTING detected")
        else:
            print(f"  ✓ Good generalization")
        
        # Fairness findings
        if data['fairness']:
            metric_frame, disparities = data['fairness']
            print(f"Fairness Assessment:")
            print(f"  Significant disparities detected: {len(disparities[disparities > 0.1])}")
            if len(disparities[disparities > 0.1]) > 0:
                print(f"  ⚠️  BIAS DETECTED - Model shows unfair treatment across groups")
            else:
                print(f"  ✓ No significant bias detected")
        else:
            print(f"Fairness Assessment: Not performed (insufficient data)")
    
    print(f"\n--- RECOMMENDATIONS ---")
    print("1. If high variance detected: Increase sample size or use stratified sampling")
    print("2. If overfitting detected: Use regularization or reduce model complexity")
    print("3. If underfitting detected: Increase model complexity or feature engineering")
    print("4. If bias detected: Implement fairness constraints or bias mitigation techniques")
    print("5. Consider using Fairlearn's mitigation strategies for biased models")

if __name__ == "__main__":
    main()
