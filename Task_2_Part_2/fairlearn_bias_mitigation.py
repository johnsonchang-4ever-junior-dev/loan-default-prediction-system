#!/usr/bin/env python3
"""
Fairlearn Bias Mitigation Implementation
Demonstrates bias detection and mitigation using Microsoft Fairlearn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Fairlearn imports
try:
    from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, false_negative_rate
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.preprocessing import CorrelationRemover
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    print("✓ Fairlearn imported successfully")
except ImportError:
    print("❌ Installing Fairlearn...")
    import subprocess
    subprocess.check_call(["pip", "install", "fairlearn"])
    from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, false_negative_rate
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.preprocessing import CorrelationRemover
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    print("✓ Fairlearn installed and imported")

def prepare_loan_data_for_fairness():
    """Prepare loan data specifically for fairness analysis"""
    print("=== PREPARING LOAN DATA FOR FAIRNESS ANALYSIS ===\n")
    
    # Load datasets from CSV_dataset directory
    try:
        approval_df = pd.read_csv('CSV_dataset/Approval.csv')
        print(f"✓ Loaded approval dataset: {approval_df.shape}")
    except Exception as e:
        print(f"Error loading approval dataset: {e}")
        return None
    
    # Clean and prepare data
    df = approval_df.copy()
    
    # Remove RiskScore if present
    riskscore_cols = [col for col in df.columns if 'risk' in col.lower() and 'score' in col.lower()]
    if riskscore_cols:
        df = df.drop(columns=riskscore_cols)
        print(f"Removed RiskScore columns: {riskscore_cols}")
    
    # Create age groups for fairness analysis
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 30, 40, 50, 100], 
                            labels=['Young', 'Middle', 'Senior', 'Elderly'])
        print("Created Age_Group categories: Young, Middle, Senior, Elderly")
    
    # Prepare sensitive features
    sensitive_features = []
    if 'Age_Group' in df.columns:
        sensitive_features.append('Age_Group')
    if 'MaritalStatus' in df.columns:
        sensitive_features.append('MaritalStatus')
    if 'EducationLevel' in df.columns:
        sensitive_features.append('EducationLevel')
    if 'EmploymentStatus' in df.columns:
        sensitive_features.append('EmploymentStatus')
    
    print(f"Sensitive features identified: {sensitive_features}")
    
    # Prepare features and target
    target_col = 'LoanApproved'
    if target_col not in df.columns:
        print(f"Target column {target_col} not found!")
        return None
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Create binary sensitive feature for initial analysis
    # Use Age_Group as primary sensitive feature
    if 'Age_Group' in X.columns:
        # Create binary age groups: Young+Middle vs Senior+Elderly
        X['Age_Binary'] = X['Age_Group'].isin(['Young', 'Middle']).astype(int)
        sensitive_feature = 'Age_Binary'
        print("Created binary age groups: Young+Middle (1) vs Senior+Elderly (0)")
    else:
        # Fallback to other sensitive features
        if 'MaritalStatus' in X.columns:
            # Create binary marital status
            X['Marital_Binary'] = (X['MaritalStatus'] == 'Married').astype(int)
            sensitive_feature = 'Marital_Binary'
        else:
            print("No suitable sensitive feature found")
            return None
    
    # Prepare features for training (exclude sensitive features)
    feature_cols = [col for col in X.columns if col not in sensitive_features + [sensitive_feature]]
    X_features = X[feature_cols]
    
    # Handle categorical variables
    for col in X_features.select_dtypes(include=['object']).columns:
        X_features = pd.get_dummies(X_features, columns=[col], drop_first=True)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_features_imputed = pd.DataFrame(imputer.fit_transform(X_features), columns=X_features.columns)
    
    print(f"Final feature matrix: {X_features_imputed.shape}")
    print(f"Sensitive feature: {sensitive_feature}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Sensitive feature distribution: {X[sensitive_feature].value_counts().to_dict()}")
    
    return X_features_imputed, y, X[sensitive_feature], sensitive_feature

def assess_baseline_fairness(X, y, sensitive_features, sensitive_feature_name):
    """Assess baseline fairness of the original model"""
    print(f"\n=== BASELINE FAIRNESS ASSESSMENT ===")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get sensitive features for test set
    sensitive_train = sensitive_features.iloc[X_train.index]
    sensitive_test = sensitive_features.iloc[X_test.index]
    
    # Train baseline model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate fairness metrics
    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'selection_rate': selection_rate,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate
    }
    
    # Create MetricFrame
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_test
    )
    
    print(f"Baseline Model Performance by {sensitive_feature_name}:")
    print(metric_frame.by_group)
    
    # Calculate disparities
    disparities = metric_frame.difference()
    print(f"\nDisparities (max - min):")
    print(disparities)
    
    # Identify significant disparities
    significant_metrics = []
    for metric in ['precision', 'recall', 'selection_rate']:
        if metric in disparities.columns:
            max_diff = disparities[metric].max()
            if max_diff > 0.1:  # 10% threshold
                significant_metrics.append((metric, max_diff))
                print(f"⚠️  SIGNIFICANT DISPARITY in {metric}: {max_diff:.4f}")
    
    return model, metric_frame, disparities, significant_metrics

def apply_preprocessing_mitigation(X, y, sensitive_features, sensitive_feature_name):
    """Apply preprocessing bias mitigation using CorrelationRemover"""
    print(f"\n=== PREPROCESSING MITIGATION (CorrelationRemover) ===")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get sensitive features for test set
    sensitive_train = sensitive_features.iloc[X_train.index]
    sensitive_test = sensitive_features.iloc[X_test.index]
    
    # Apply CorrelationRemover
    correlation_remover = CorrelationRemover(sensitive_feature_ids=[0])  # Assuming first column is sensitive
    X_train_mitigated = correlation_remover.fit_transform(X_train)
    X_test_mitigated = correlation_remover.transform(X_test)
    
    # Train model on mitigated data
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_mitigated, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test_mitigated)
    
    # Calculate fairness metrics
    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'selection_rate': selection_rate,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate
    }
    
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_test
    )
    
    print(f"Preprocessing Mitigation Performance by {sensitive_feature_name}:")
    print(metric_frame.by_group)
    
    disparities = metric_frame.difference()
    print(f"\nDisparities after preprocessing mitigation:")
    print(disparities)
    
    return model, metric_frame, disparities

def apply_postprocessing_mitigation(X, y, sensitive_features, sensitive_feature_name):
    """Apply postprocessing bias mitigation using ThresholdOptimizer"""
    print(f"\n=== POSTPROCESSING MITIGATION (ThresholdOptimizer) ===")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get sensitive features for test set
    sensitive_train = sensitive_features.iloc[X_train.index]
    sensitive_test = sensitive_features.iloc[X_test.index]
    
    # Train base model
    base_model = LogisticRegression(random_state=42, max_iter=1000)
    base_model.fit(X_train, y_train)
    
    # Get prediction probabilities
    y_pred_proba = base_model.predict_proba(X_test)[:, 1]
    
    # Apply ThresholdOptimizer for demographic parity
    threshold_optimizer = ThresholdOptimizer(
        estimator=base_model,
        constraints="demographic_parity",
        prefit=True
    )
    
    threshold_optimizer.fit(X_test, y_test, sensitive_features=sensitive_test)
    y_pred_mitigated = threshold_optimizer.predict(X_test, sensitive_features=sensitive_test)
    
    # Calculate fairness metrics
    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'selection_rate': selection_rate,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate
    }
    
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred_mitigated,
        sensitive_features=sensitive_test
    )
    
    print(f"Postprocessing Mitigation Performance by {sensitive_feature_name}:")
    print(metric_frame.by_group)
    
    disparities = metric_frame.difference()
    print(f"\nDisparities after postprocessing mitigation:")
    print(disparities)
    
    return threshold_optimizer, metric_frame, disparities

def apply_reductions_mitigation(X, y, sensitive_features, sensitive_feature_name):
    """Apply reductions bias mitigation using ExponentiatedGradient"""
    print(f"\n=== REDUCTIONS MITIGATION (ExponentiatedGradient) ===")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get sensitive features for test set
    sensitive_train = sensitive_features.iloc[X_train.index]
    sensitive_test = sensitive_features.iloc[X_test.index]
    
    # Create base estimator
    base_estimator = LogisticRegression(random_state=42, max_iter=1000)
    
    # Apply ExponentiatedGradient with demographic parity constraint
    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=DemographicParity()
    )
    
    # Fit the mitigator
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
    
    # Get predictions
    y_pred_mitigated = mitigator.predict(X_test, sensitive_features=sensitive_test)
    
    # Calculate fairness metrics
    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'selection_rate': selection_rate,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate
    }
    
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred_mitigated,
        sensitive_features=sensitive_test
    )
    
    print(f"Reductions Mitigation Performance by {sensitive_feature_name}:")
    print(metric_frame.by_group)
    
    disparities = metric_frame.difference()
    print(f"\nDisparities after reductions mitigation:")
    print(disparities)
    
    return mitigator, metric_frame, disparities

def compare_mitigation_strategies(baseline_disparities, preprocessing_disparities, 
                                postprocessing_disparities, reductions_disparities):
    """Compare effectiveness of different mitigation strategies"""
    print(f"\n=== MITIGATION STRATEGY COMPARISON ===")
    
    strategies = {
        'Baseline': baseline_disparities,
        'Preprocessing': preprocessing_disparities,
        'Postprocessing': postprocessing_disparities,
        'Reductions': reductions_disparities
    }
    
    # Compare key metrics
    key_metrics = ['precision', 'recall', 'selection_rate']
    
    print(f"{'Strategy':<15} {'Precision':<12} {'Recall':<12} {'Selection Rate':<15}")
    print("-" * 60)
    
    for strategy_name, disparities in strategies.items():
        if disparities is not None:
            precision_diff = disparities.get('precision', {}).max() if 'precision' in disparities.columns else 0
            recall_diff = disparities.get('recall', {}).max() if 'recall' in disparities.columns else 0
            selection_diff = disparities.get('selection_rate', {}).max() if 'selection_rate' in disparities.columns else 0
            
            print(f"{strategy_name:<15} {precision_diff:<12.4f} {recall_diff:<12.4f} {selection_diff:<15.4f}")
    
    # Identify best strategy for each metric
    print(f"\n--- Best Strategy by Metric ---")
    for metric in key_metrics:
        if metric in baseline_disparities.columns:
            best_strategy = min(strategies.keys(), 
                              key=lambda x: strategies[x].get(metric, {}).max() 
                              if strategies[x] is not None and metric in strategies[x].columns else float('inf'))
            best_value = strategies[best_strategy].get(metric, {}).max()
            print(f"{metric}: {best_strategy} (disparity: {best_value:.4f})")

def create_fairness_comparison_plot(baseline_metrics, preprocessing_metrics, 
                                  postprocessing_metrics, reductions_metrics, sensitive_feature_name):
    """Create visualization comparing different mitigation strategies"""
    print(f"\n--- Creating Fairness Comparison Visualization ---")
    
    strategies = ['Baseline', 'Preprocessing', 'Postprocessing', 'Reductions']
    metric_frames = [baseline_metrics, preprocessing_metrics, postprocessing_metrics, reductions_metrics]
    
    # Extract selection rates for each group
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Fairness Mitigation Comparison by {sensitive_feature_name}', fontsize=16)
    
    # Plot 1: Selection Rate by Group
    for i, (strategy, metric_frame) in enumerate(zip(strategies, metric_frames)):
        if metric_frame is not None:
            selection_rates = metric_frame.by_group['selection_rate']
            axes[0, 0].bar([f"{strategy}\n{group}" for group in selection_rates.index], 
                          selection_rates.values, alpha=0.7, label=strategy)
    
    axes[0, 0].set_title('Selection Rate by Group')
    axes[0, 0].set_ylabel('Selection Rate')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Precision by Group
    for i, (strategy, metric_frame) in enumerate(zip(strategies, metric_frames)):
        if metric_frame is not None:
            precision_rates = metric_frame.by_group['precision']
            axes[0, 1].bar([f"{strategy}\n{group}" for group in precision_rates.index], 
                          precision_rates.values, alpha=0.7, label=strategy)
    
    axes[0, 1].set_title('Precision by Group')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Recall by Group
    for i, (strategy, metric_frame) in enumerate(zip(strategies, metric_frames)):
        if metric_frame is not None:
            recall_rates = metric_frame.by_group['recall']
            axes[1, 0].bar([f"{strategy}\n{group}" for group in recall_rates.index], 
                          recall_rates.values, alpha=0.7, label=strategy)
    
    axes[1, 0].set_title('Recall by Group')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: F1 Score by Group
    for i, (strategy, metric_frame) in enumerate(zip(strategies, metric_frames)):
        if metric_frame is not None:
            f1_rates = metric_frame.by_group['f1']
            axes[1, 1].bar([f"{strategy}\n{group}" for group in f1_rates.index], 
                          f1_rates.values, alpha=0.7, label=strategy)
    
    axes[1, 1].set_title('F1 Score by Group')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'fairness_mitigation_comparison_{sensitive_feature_name.lower()}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Fairness comparison plot saved as: {plot_filename}")
    
    plt.show()

def main():
    """Main function to demonstrate Fairlearn bias mitigation"""
    print("🔍 FAIRLEARN BIAS MITIGATION DEMONSTRATION")
    print("=" * 60)
    
    # Prepare data
    data = prepare_loan_data_for_fairness()
    if data is None:
        print("❌ Failed to prepare data")
        return
    
    X, y, sensitive_features, sensitive_feature_name = data
    
    # 1. Assess baseline fairness
    baseline_model, baseline_metrics, baseline_disparities, significant_metrics = assess_baseline_fairness(
        X, y, sensitive_features, sensitive_feature_name
    )
    
    if not significant_metrics:
        print("✓ No significant bias detected in baseline model")
        print("Proceeding with mitigation demonstration anyway...")
    
    # 2. Apply preprocessing mitigation
    preprocessing_model, preprocessing_metrics, preprocessing_disparities = apply_preprocessing_mitigation(
        X, y, sensitive_features, sensitive_feature_name
    )
    
    # 3. Apply postprocessing mitigation
    postprocessing_model, postprocessing_metrics, postprocessing_disparities = apply_postprocessing_mitigation(
        X, y, sensitive_features, sensitive_feature_name
    )
    
    # 4. Apply reductions mitigation
    reductions_model, reductions_metrics, reductions_disparities = apply_reductions_mitigation(
        X, y, sensitive_features, sensitive_feature_name
    )
    
    # 5. Compare strategies
    compare_mitigation_strategies(
        baseline_disparities, preprocessing_disparities, 
        postprocessing_disparities, reductions_disparities
    )
    
    # 6. Create comparison visualization
    create_fairness_comparison_plot(
        baseline_metrics, preprocessing_metrics, 
        postprocessing_metrics, reductions_metrics, sensitive_feature_name
    )
    
    print(f"\n✅ Fairlearn bias mitigation demonstration completed!")
    print("📁 Check the generated visualizations for detailed comparison results.")

if __name__ == "__main__":
    main()
