#!/usr/bin/env python3
"""
Comprehensive Bias Analysis Report Generator
Creates a detailed report for Task 2 Part 2 covering all required aspects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def generate_comprehensive_bias_report():
    """Generate comprehensive bias analysis report for Task 2 Part 2"""
    
    print("📊 COMPREHENSIVE BIAS ANALYSIS REPORT")
    print("=" * 80)
    print("Task 2 Part 2: Deliberation on Task 1 - Ethics and Bias Analysis")
    print("=" * 80)
    
    # Load datasets from CSV_dataset directory
    try:
        approval_df = pd.read_csv('CSV_dataset/Approval.csv')
        payment_df = pd.read_csv('CSV_dataset/Repayment.csv')
        print(f"✓ Loaded datasets: Approval {approval_df.shape}, Payment {payment_df.shape}")
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return
    
    # Generate report sections
    report_sections = []
    
    # Section 1: Cross-validation and Sampling Bias Analysis
    report_sections.append(analyze_cross_validation_bias(approval_df, payment_df))
    
    # Section 2: Learning Curve Analysis
    report_sections.append(analyze_learning_curves(approval_df, payment_df))
    
    # Section 3: Data Balancing Bias Analysis
    report_sections.append(analyze_data_balancing_bias(approval_df, payment_df))
    
    # Section 4: Demographic Bias Analysis
    report_sections.append(analyze_demographic_bias(approval_df, payment_df))
    
    # Section 5: Security, Privacy, and Ethical Risks
    report_sections.append(analyze_ethical_risks())
    
    # Section 6: Recommendations and Mitigation Strategies
    report_sections.append(generate_recommendations())
    
    # Compile final report
    compile_final_report(report_sections)

def analyze_cross_validation_bias(approval_df, payment_df):
    """Analyze cross-validation and sampling strategies for bias"""
    print("\n" + "="*60)
    print("1. CROSS-VALIDATION AND SAMPLING BIAS ANALYSIS")
    print("="*60)
    
    analysis = {
        'title': 'Cross-Validation and Sampling Bias Analysis',
        'findings': [],
        'recommendations': []
    }
    
    datasets = [
        (approval_df, 'LoanApproved', 'Approval'),
        (payment_df, 'fully.paid', 'Payment')
    ]
    
    for df, target_col, dataset_name in datasets:
        print(f"\n--- {dataset_name} Dataset Analysis ---")
        
        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # 5-Fold Cross-Validation Analysis
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)
        }
        
        cv_results = {}
        for model_name, model in models.items():
            # Standard CV
            cv_scores = cross_val_score(model, X_imputed, y, cv=5, scoring='precision')
            
            # Stratified CV
            stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            stratified_scores = cross_val_score(model, X_imputed, y, cv=stratified_cv, scoring='precision')
            
            cv_results[model_name] = {
                'standard_cv': cv_scores,
                'stratified_cv': stratified_scores
            }
            
            print(f"\n{model_name} - {dataset_name}:")
            print(f"  Standard CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  Stratified CV: {stratified_scores.mean():.4f} ± {stratified_scores.std():.4f}")
            
            # Check for sampling bias
            standard_std = cv_scores.std()
            stratified_std = stratified_scores.std()
            
            if standard_std > 0.05:
                analysis['findings'].append(f"⚠️ {dataset_name} {model_name}: High variance in standard CV (std={standard_std:.4f}) - potential sampling bias")
            else:
                analysis['findings'].append(f"✓ {dataset_name} {model_name}: Low variance in standard CV (std={standard_std:.4f}) - good sampling stability")
            
            if abs(standard_std - stratified_std) > 0.02:
                analysis['findings'].append(f"📊 {dataset_name} {model_name}: Stratified CV shows different variance - class imbalance affects sampling")
    
    # Sampling Strategy Assessment
    print(f"\n--- Sampling Strategy Assessment ---")
    
    # Check for class imbalance
    approval_imbalance = approval_df['LoanApproved'].value_counts(normalize=True)
    payment_imbalance = payment_df['fully.paid'].value_counts(normalize=True)
    
    print(f"Approval dataset imbalance: {approval_imbalance.to_dict()}")
    print(f"Payment dataset imbalance: {payment_imbalance.to_dict()}")
    
    if approval_imbalance.min() < 0.2:
        analysis['findings'].append("⚠️ Approval dataset: Severe class imbalance - may cause sampling bias")
        analysis['recommendations'].append("Use stratified sampling for approval dataset to ensure balanced representation")
    
    if payment_imbalance.min() < 0.2:
        analysis['findings'].append("⚠️ Payment dataset: Severe class imbalance - may cause sampling bias")
        analysis['recommendations'].append("Use stratified sampling for payment dataset to ensure balanced representation")
    
    # Random State Analysis
    print(f"\n--- Random State Sensitivity Analysis ---")
    random_states = [42, 123, 456, 789, 999]
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    for dataset_name, df, target_col in [('Approval', approval_df, 'LoanApproved'), ('Payment', payment_df, 'fully.paid')]:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        scores_by_random_state = []
        for rs in random_states:
            X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=rs, stratify=y)
            model.fit(X_train, y_train)
            score = precision_score(y_test, model.predict(X_test))
            scores_by_random_state.append(score)
        
        score_std = np.std(scores_by_random_state)
        print(f"{dataset_name} - Score std across random states: {score_std:.4f}")
        
        if score_std > 0.02:
            analysis['findings'].append(f"⚠️ {dataset_name}: High sensitivity to random state (std={score_std:.4f}) - unstable sampling")
        else:
            analysis['findings'].append(f"✓ {dataset_name}: Low sensitivity to random state (std={score_std:.4f}) - stable sampling")
    
    return analysis

def analyze_learning_curves(approval_df, payment_df):
    """Analyze learning curves for different training set sizes"""
    print("\n" + "="*60)
    print("2. LEARNING CURVE ANALYSIS")
    print("="*60)
    
    analysis = {
        'title': 'Learning Curve Analysis',
        'findings': [],
        'recommendations': []
    }
    
    datasets = [
        (approval_df, 'LoanApproved', 'Approval'),
        (payment_df, 'fully.paid', 'Payment')
    ]
    
    for df, target_col, dataset_name in datasets:
        print(f"\n--- {dataset_name} Dataset Learning Curves ---")
        
        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)
        }
        
        for model_name, model in models.items():
            print(f"\n{model_name} - {dataset_name}:")
            
            # Generate learning curve
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X_imputed, y, train_sizes=train_sizes, cv=5, 
                scoring='precision', n_jobs=-1, random_state=42
            )
            
            # Calculate statistics
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Analyze learning curve patterns
            final_train_score = train_mean[-1]
            final_val_score = val_mean[-1]
            gap = final_train_score - final_val_score
            convergence_point = None
            
            # Find convergence point (where validation score stabilizes)
            val_scores_smoothed = np.convolve(val_mean, np.ones(3)/3, mode='valid')
            if len(val_scores_smoothed) > 2:
                val_diff = np.diff(val_scores_smoothed)
                convergence_idx = np.where(np.abs(val_diff) < 0.01)[0]
                if len(convergence_idx) > 0:
                    convergence_point = train_sizes_abs[convergence_idx[0] + 1]
            
            print(f"  Final Training Score: {final_train_score:.4f}")
            print(f"  Final Validation Score: {final_val_score:.4f}")
            print(f"  Generalization Gap: {gap:.4f}")
            if convergence_point:
                print(f"  Convergence Point: {convergence_point:.0f} samples")
            
            # Assess learning curve patterns
            if gap > 0.15:
                analysis['findings'].append(f"⚠️ {dataset_name} {model_name}: Severe overfitting (gap={gap:.4f})")
                analysis['recommendations'].append(f"Apply regularization to {model_name} for {dataset_name} dataset")
            elif gap > 0.1:
                analysis['findings'].append(f"⚠️ {dataset_name} {model_name}: Moderate overfitting (gap={gap:.4f})")
                analysis['recommendations'].append(f"Consider reducing model complexity for {model_name} in {dataset_name} dataset")
            elif final_val_score < 0.6:
                analysis['findings'].append(f"⚠️ {dataset_name} {model_name}: Underfitting (val_score={final_val_score:.4f})")
                analysis['recommendations'].append(f"Increase model complexity or add features for {model_name} in {dataset_name} dataset")
            else:
                analysis['findings'].append(f"✓ {dataset_name} {model_name}: Good generalization (gap={gap:.4f})")
            
            # Check if more data would help
            if len(val_scores) > 3:
                recent_val_trend = val_mean[-3:].mean() - val_mean[-6:-3].mean() if len(val_mean) >= 6 else 0
                if recent_val_trend > 0.01:
                    analysis['findings'].append(f"📈 {dataset_name} {model_name}: Validation score still improving - more data may help")
                    analysis['recommendations'].append(f"Consider collecting more data for {dataset_name} dataset")
                elif recent_val_trend < -0.01:
                    analysis['findings'].append(f"📉 {dataset_name} {model_name}: Validation score declining - possible overfitting")
    
    return analysis

def analyze_data_balancing_bias(approval_df, payment_df):
    """Analyze bias introduced by data balancing techniques"""
    print("\n" + "="*60)
    print("3. DATA BALANCING BIAS ANALYSIS")
    print("="*60)
    
    analysis = {
        'title': 'Data Balancing Bias Analysis',
        'findings': [],
        'recommendations': []
    }
    
    # Check if balanced datasets exist
    try:
        approval_balanced = pd.read_csv('Loan_APorNAP2_ohe_balanced_50_50.csv')
        payment_balanced = pd.read_csv('Loan_PAorNPA2_ohe_balanced_50_50.csv')
        print("✓ Found balanced datasets")
    except FileNotFoundError:
        print("⚠️ Balanced datasets not found - analyzing original datasets only")
        return analysis
    
    datasets = [
        (approval_df, approval_balanced, 'LoanApproved', 'Approval'),
        (payment_df, payment_balanced, 'fully.paid', 'Payment')
    ]
    
    for original_df, balanced_df, target_col, dataset_name in datasets:
        print(f"\n--- {dataset_name} Dataset Balancing Analysis ---")
        
        # Compare target distributions
        original_dist = original_df[target_col].value_counts(normalize=True)
        balanced_dist = balanced_df[target_col].value_counts(normalize=True)
        
        print(f"Original distribution: {original_dist.to_dict()}")
        print(f"Balanced distribution: {balanced_dist.to_dict()}")
        
        # Calculate balancing impact
        minority_class = original_dist.idxmin()
        minority_original = original_dist[minority_class]
        minority_balanced = balanced_dist[minority_class]
        
        balancing_ratio = minority_balanced / minority_original
        print(f"Minority class balancing ratio: {balancing_ratio:.2f}x")
        
        if balancing_ratio > 4:
            analysis['findings'].append(f"⚠️ {dataset_name}: High balancing ratio ({balancing_ratio:.1f}x) - may introduce overfitting")
            analysis['recommendations'].append(f"Consider using SMOTE or other synthetic generation for {dataset_name} instead of duplication")
        elif balancing_ratio > 2:
            analysis['findings'].append(f"📊 {dataset_name}: Moderate balancing ratio ({balancing_ratio:.1f}x) - monitor for overfitting")
        else:
            analysis['findings'].append(f"✓ {dataset_name}: Reasonable balancing ratio ({balancing_ratio:.1f}x)")
        
        # Analyze sensitive feature distribution changes
        sensitive_features = ['Age', 'MaritalStatus', 'EducationLevel', 'EmploymentStatus']
        available_sensitive = [col for col in sensitive_features if col in original_df.columns]
        
        if available_sensitive:
            print(f"\nSensitive Feature Distribution Changes:")
            for col in available_sensitive:
                if col in balanced_df.columns:
                    # Calculate distribution change
                    orig_dist = original_df[col].value_counts(normalize=True)
                    bal_dist = balanced_df[col].value_counts(normalize=True)
                    
                    # Calculate KL divergence
                    from scipy.stats import entropy
                    kl_div = entropy(bal_dist, orig_dist)
                    
                    print(f"  {col}: KL divergence = {kl_div:.4f}")
                    
                    if kl_div > 0.2:
                        analysis['findings'].append(f"⚠️ {dataset_name}: Significant change in {col} distribution (KL={kl_div:.4f})")
                        analysis['recommendations'].append(f"Monitor {col} bias in {dataset_name} model predictions")
                    elif kl_div > 0.1:
                        analysis['findings'].append(f"📊 {dataset_name}: Moderate change in {col} distribution (KL={kl_div:.4f})")
                    else:
                        analysis['findings'].append(f"✓ {dataset_name}: {col} distribution preserved (KL={kl_div:.4f})")
        
        # Check for exact duplicates
        minority_original_data = original_df[original_df[target_col] == minority_class]
        minority_balanced_data = balanced_df[balanced_df[target_col] == minority_class]
        
        # Sample check for duplicates
        if len(minority_balanced_data) > len(minority_original_data):
            sample_size = min(100, len(minority_original_data))
            original_sample = minority_original_data.sample(n=sample_size, random_state=42)
            
            # Check how many of the balanced data are exact duplicates
            duplicate_count = 0
            for _, row in original_sample.iterrows():
                if len(balanced_df[(balanced_df == row).all(axis=1)]) > 1:
                    duplicate_count += 1
            
            duplicate_ratio = duplicate_count / sample_size
            print(f"Duplicate ratio in minority class: {duplicate_ratio:.2%}")
            
            if duplicate_ratio > 0.5:
                analysis['findings'].append(f"⚠️ {dataset_name}: High duplicate ratio ({duplicate_ratio:.1%}) - may cause overfitting")
                analysis['recommendations'].append(f"Use synthetic data generation instead of duplication for {dataset_name}")
            elif duplicate_ratio > 0.2:
                analysis['findings'].append(f"📊 {dataset_name}: Moderate duplicate ratio ({duplicate_ratio:.1%}) - monitor model performance")
            else:
                analysis['findings'].append(f"✓ {dataset_name}: Low duplicate ratio ({duplicate_ratio:.1%})")
    
    return analysis

def analyze_demographic_bias(approval_df, payment_df):
    """Analyze demographic bias in the datasets and models"""
    print("\n" + "="*60)
    print("4. DEMOGRAPHIC BIAS ANALYSIS")
    print("="*60)
    
    analysis = {
        'title': 'Demographic Bias Analysis',
        'findings': [],
        'recommendations': []
    }
    
    datasets = [
        (approval_df, 'LoanApproved', 'Approval'),
        (payment_df, 'fully.paid', 'Payment')
    ]
    
    for df, target_col, dataset_name in datasets:
        print(f"\n--- {dataset_name} Dataset Demographic Analysis ---")
        
        # Identify sensitive features
        sensitive_features = []
        if 'Age' in df.columns:
            # Create age groups
            df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], labels=['Young', 'Middle', 'Senior', 'Elderly'])
            sensitive_features.append('Age_Group')
        
        if 'MaritalStatus' in df.columns:
            sensitive_features.append('MaritalStatus')
        
        if 'EducationLevel' in df.columns:
            sensitive_features.append('EducationLevel')
        
        if 'EmploymentStatus' in df.columns:
            sensitive_features.append('EmploymentStatus')
        
        print(f"Sensitive features: {sensitive_features}")
        
        # Analyze target distribution by sensitive features
        for feature in sensitive_features:
            if feature in df.columns:
                print(f"\n{feature} distribution by {target_col}:")
                crosstab = pd.crosstab(df[feature], df[target_col], normalize='index')
                print(crosstab)
                
                # Calculate disparity
                if len(crosstab.columns) == 2:  # Binary target
                    positive_class = crosstab.columns[1]  # Assuming 1 is positive
                    approval_rates = crosstab[positive_class]
                    
                    max_rate = approval_rates.max()
                    min_rate = approval_rates.min()
                    disparity = max_rate - min_rate
                    
                    print(f"Approval rate disparity: {disparity:.4f} (max: {max_rate:.4f}, min: {min_rate:.4f})")
                    
                    if disparity > 0.2:
                        analysis['findings'].append(f"⚠️ {dataset_name}: High disparity in {feature} ({disparity:.4f})")
                        analysis['recommendations'].append(f"Investigate {feature} bias in {dataset_name} model")
                    elif disparity > 0.1:
                        analysis['findings'].append(f"📊 {dataset_name}: Moderate disparity in {feature} ({disparity:.4f})")
                    else:
                        analysis['findings'].append(f"✓ {dataset_name}: Low disparity in {feature} ({disparity:.4f})")
        
        # Analyze feature importance bias
        print(f"\n--- Feature Importance Analysis ---")
        
        # Prepare data for model training
        X = df.drop(columns=[target_col] + sensitive_features)
        y = df[target_col]
        
        # Handle categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Train Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_imputed, y)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 most important features:")
        print(feature_importance.head(10))
        
        # Check for potentially biased features
        potentially_biased = []
        for feature in feature_importance['feature'].head(10):
            if any(sensitive in feature.lower() for sensitive in ['age', 'gender', 'race', 'marital', 'education']):
                potentially_biased.append(feature)
        
        if potentially_biased:
            analysis['findings'].append(f"⚠️ {dataset_name}: Potentially biased features in top 10: {potentially_biased}")
            analysis['recommendations'].append(f"Review feature selection for {dataset_name} to avoid proxy discrimination")
        else:
            analysis['findings'].append(f"✓ {dataset_name}: No obviously biased features in top 10")
    
    return analysis

def analyze_ethical_risks():
    """Analyze security, privacy, and ethical risks"""
    print("\n" + "="*60)
    print("5. SECURITY, PRIVACY, AND ETHICAL RISKS ANALYSIS")
    print("="*60)
    
    analysis = {
        'title': 'Security, Privacy, and Ethical Risks Analysis',
        'findings': [],
        'recommendations': []
    }
    
    # Security Risks
    print("\n--- Security Risks ---")
    security_risks = [
        "Model inversion attacks could reveal sensitive training data",
        "Adversarial attacks could manipulate loan decisions",
        "Model stealing could compromise proprietary algorithms",
        "Data poisoning could introduce bias in future training"
    ]
    
    for risk in security_risks:
        analysis['findings'].append(f"🔒 Security Risk: {risk}")
    
    analysis['recommendations'].extend([
        "Implement differential privacy techniques",
        "Use adversarial training to improve robustness",
        "Deploy model monitoring for unusual prediction patterns",
        "Implement access controls and audit logging"
    ])
    
    # Privacy Risks
    print("\n--- Privacy Risks ---")
    privacy_risks = [
        "Personal financial data could be exposed in model outputs",
        "Inference attacks could reveal individual loan applications",
        "Data aggregation could lead to re-identification",
        "Model updates could leak information about new data"
    ]
    
    for risk in privacy_risks:
        analysis['findings'].append(f"🔐 Privacy Risk: {risk}")
    
    analysis['recommendations'].extend([
        "Implement data anonymization techniques",
        "Use federated learning to keep data local",
        "Apply k-anonymity and l-diversity principles",
        "Implement data minimization strategies"
    ])
    
    # Ethical Risks
    print("\n--- Ethical Risks ---")
    ethical_risks = [
        "Algorithmic discrimination against protected groups",
        "Perpetuation of historical biases in lending",
        "Lack of transparency in decision-making process",
        "Unfair denial of credit to qualified applicants",
        "Reinforcement of socioeconomic inequalities"
    ]
    
    for risk in ethical_risks:
        analysis['findings'].append(f"⚖️ Ethical Risk: {risk}")
    
    analysis['recommendations'].extend([
        "Implement fairness constraints in model training",
        "Regular bias auditing and monitoring",
        "Provide explanations for loan decisions",
        "Establish appeal processes for denied applications",
        "Diverse team involvement in model development"
    ])
    
    # Scale Deployment Risks
    print("\n--- Large-Scale Deployment Risks ---")
    scale_risks = [
        "Systemic bias amplification across large populations",
        "Regulatory compliance challenges in multiple jurisdictions",
        "Increased attack surface for malicious actors",
        "Resource consumption and environmental impact",
        "Dependency on external data sources and APIs"
    ]
    
    for risk in scale_risks:
        analysis['findings'].append(f"📈 Scale Risk: {risk}")
    
    analysis['recommendations'].extend([
        "Implement comprehensive monitoring and alerting",
        "Establish governance frameworks for model updates",
        "Regular compliance audits and reporting",
        "Sustainable computing practices",
        "Robust backup and disaster recovery plans"
    ])
    
    return analysis

def generate_recommendations():
    """Generate comprehensive recommendations"""
    print("\n" + "="*60)
    print("6. RECOMMENDATIONS AND MITIGATION STRATEGIES")
    print("="*60)
    
    analysis = {
        'title': 'Recommendations and Mitigation Strategies',
        'findings': [],
        'recommendations': []
    }
    
    # Technical Recommendations
    print("\n--- Technical Recommendations ---")
    technical_recs = [
        "Implement stratified cross-validation to ensure balanced representation",
        "Use ensemble methods to reduce overfitting from data balancing",
        "Apply regularization techniques (L1/L2) to prevent overfitting",
        "Implement feature selection to remove potentially biased features",
        "Use synthetic data generation (SMOTE) instead of simple duplication",
        "Deploy model monitoring for performance drift and bias detection",
        "Implement A/B testing for model updates",
        "Use interpretable models where possible for transparency"
    ]
    
    for rec in technical_recs:
        analysis['recommendations'].append(f"🔧 Technical: {rec}")
    
    # Fairness Recommendations
    print("\n--- Fairness Recommendations ---")
    fairness_recs = [
        "Implement demographic parity constraints using Fairlearn",
        "Use equalized odds to ensure similar error rates across groups",
        "Regular bias auditing using multiple fairness metrics",
        "Diverse dataset collection and validation",
        "Intersectional analysis for multiple protected attributes",
        "Human-in-the-loop validation for edge cases",
        "Bias mitigation through preprocessing and postprocessing",
        "Regular retraining with updated fairness constraints"
    ]
    
    for rec in fairness_recs:
        analysis['recommendations'].append(f"⚖️ Fairness: {rec}")
    
    # Governance Recommendations
    print("\n--- Governance Recommendations ---")
    governance_recs = [
        "Establish AI ethics committee with diverse representation",
        "Implement model versioning and rollback capabilities",
        "Regular third-party bias audits and assessments",
        "Clear documentation of model limitations and assumptions",
        "User education about algorithmic decision-making",
        "Transparent communication about data usage and model updates",
        "Regular stakeholder feedback and model improvement cycles",
        "Compliance with relevant regulations (GDPR, CCPA, etc.)"
    ]
    
    for rec in governance_recs:
        analysis['recommendations'].append(f"🏛️ Governance: {rec}")
    
    # Monitoring Recommendations
    print("\n--- Monitoring Recommendations ---")
    monitoring_recs = [
        "Real-time bias monitoring dashboards",
        "Automated alerts for fairness metric violations",
        "Regular performance evaluation across demographic groups",
        "Data drift detection and model retraining triggers",
        "User feedback collection and analysis",
        "Regular model explainability reports",
        "Adversarial testing and robustness evaluation",
        "Continuous learning from model performance data"
    ]
    
    for rec in monitoring_recs:
        analysis['recommendations'].append(f"📊 Monitoring: {rec}")
    
    return analysis

def compile_final_report(report_sections):
    """Compile all sections into a comprehensive report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE BIAS ANALYSIS REPORT - EXECUTIVE SUMMARY")
    print("="*80)
    
    # Count findings by type
    total_findings = 0
    critical_findings = 0
    total_recommendations = 0
    
    for section in report_sections:
        total_findings += len(section['findings'])
        critical_findings += len([f for f in section['findings'] if '⚠️' in f])
        total_recommendations += len(section['recommendations'])
    
    print(f"\n📊 REPORT SUMMARY:")
    print(f"  Total Findings: {total_findings}")
    print(f"  Critical Issues: {critical_findings}")
    print(f"  Recommendations: {total_recommendations}")
    
    # Print each section
    for i, section in enumerate(report_sections, 1):
        print(f"\n{i}. {section['title'].upper()}")
        print("-" * 60)
        
        if section['findings']:
            print("\nKey Findings:")
            for finding in section['findings']:
                print(f"  {finding}")
        
        if section['recommendations']:
            print("\nRecommendations:")
            for rec in section['recommendations']:
                print(f"  {rec}")
    
    # Final recommendations
    print(f"\n🎯 PRIORITY ACTIONS:")
    print("1. Implement stratified cross-validation immediately")
    print("2. Deploy bias monitoring using Fairlearn")
    print("3. Establish AI ethics governance framework")
    print("4. Conduct regular bias audits and model retraining")
    print("5. Implement transparency and explainability features")
    
    print(f"\n✅ Comprehensive bias analysis completed!")
    print("This report provides a foundation for ethical AI deployment in loan decision-making.")

if __name__ == "__main__":
    generate_comprehensive_bias_report()
