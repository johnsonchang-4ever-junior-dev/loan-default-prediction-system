# Comprehensive Loan Analysis: Top Factors, Algorithm Performance, and Business Insights
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def analyze_loan_datasets():
    """Comprehensive analysis of loan datasets to answer business questions"""
    
    print("=== COMPREHENSIVE LOAN ANALYSIS ===\n")
    
    # 1. LOAD AND EXPLORE DATASETS
    print("1. LOADING DATASETS...")
    try:
        # Load approval dataset (APorNAP2)
        loan_approval = pd.read_csv('Loan_APorNAP2.csv')
        print(f"✓ Loaded Loan_APorNAP2.csv: {loan_approval.shape}")
        
        # Load payment dataset (PAorNPA2) 
        loan_payment = pd.read_csv('Loan_PAorNPA2.csv')
        print(f"✓ Loaded Loan_PAorNPA2.csv: {loan_payment.shape}")
        
        # Load one-hot encoded version if available
        try:
            loan_payment_ohe = pd.read_csv('PAorNPA2_ohe.csv')
            print(f"✓ Loaded PAorNPA2_ohe.csv: {loan_payment_ohe.shape}")
        except:
            loan_payment_ohe = None
            print("! PAorNPA2_ohe.csv not found - will create encoding")
            
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    print(f"\n=== DATASET OVERVIEW ===")
    print(f"Approval Dataset: {loan_approval.shape[0]:,} rows, {loan_approval.shape[1]} columns")
    print(f"Payment Dataset: {loan_payment.shape[0]:,} rows, {loan_payment.shape[1]} columns")
    
    # Display target distributions
    print(f"\n=== TARGET VARIABLE DISTRIBUTIONS ===")
    
    # Approval dataset target
    approval_targets = [col for col in loan_approval.columns if any(word in col.lower() for word in ['approv', 'accept', 'loan_status'])]
    if approval_targets:
        approval_target = approval_targets[0]
        print(f"Approval Target ({approval_target}):")
        print(loan_approval[approval_target].value_counts(normalize=True).round(4))
    
    # Payment dataset target  
    payment_targets = [col for col in loan_payment.columns if any(word in col.lower() for word in ['paid', 'default'])]
    if payment_targets:
        payment_target = payment_targets[0]
        print(f"\nPayment Target ({payment_target}):")
        print(loan_payment[payment_target].value_counts(normalize=True).round(4))
        
        # Create binary version for better analysis
        if payment_target == 'PreviousLoanDefaults':
            loan_payment['HasAnyDefaults'] = (loan_payment[payment_target] > 0).astype(int)
            payment_target = 'HasAnyDefaults'
            print(f"\nBinary Payment Target ({payment_target}):")
            print(loan_payment[payment_target].value_counts(normalize=True).round(4))
    
    # 2. IDENTIFY TOP 3 FACTORS FOR LOAN APPROVAL
    print(f"\n=== 2. TOP 3 FACTORS FOR LOAN APPROVAL ===")
    
    def get_feature_importance(X, y, dataset_name):
        """Get feature importance using Random Forest"""
        # Handle missing values and prepare data
        X_clean = X.copy()
        
        # Convert categorical columns to numeric
        for col in X_clean.select_dtypes(include=['object']).columns:
            X_clean = pd.get_dummies(X_clean, columns=[col], drop_first=True)
        
        # Fill missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X_clean), columns=X_clean.columns)
        
        # Train Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_imputed, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'Feature': X_clean.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\n{dataset_name} - Top 10 Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df.head(3)
    
    # Analyze approval dataset
    if approval_targets:
        X_approval = loan_approval.drop(columns=[approval_target])
        y_approval = loan_approval[approval_target]
        top_approval_factors = get_feature_importance(X_approval, y_approval, "APPROVAL DATASET")
    
    # Analyze payment dataset
    if payment_targets:
        X_payment = loan_payment.drop(columns=[payment_target])
        y_payment = loan_payment[payment_target]
        top_payment_factors = get_feature_importance(X_payment, y_payment, "PAYMENT DATASET")
    
    # 3. ALGORITHM PERFORMANCE COMPARISON
    print(f"\n=== 3. ALGORITHM PERFORMANCE COMPARISON ===")
    
    def compare_algorithms(X, y, dataset_name):
        """Compare Logistic Regression, Decision Tree, and Random Forest"""
        
        # Prepare data
        X_clean = X.copy()
        for col in X_clean.select_dtypes(include=['object']).columns:
            X_clean = pd.get_dummies(X_clean, columns=[col], drop_first=True)
        
        # Train-test split
        # Check if stratification is possible
        if y.nunique() > 1 and y.value_counts().min() >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y, test_size=0.2, random_state=42
            )
        
        # Define models
        models = {
            'Logistic Regression': make_pipeline(SimpleImputer(strategy='median'), 
                                                StandardScaler(), 
                                                LogisticRegression(random_state=42, max_iter=1000)),
            'Decision Tree': make_pipeline(SimpleImputer(strategy='median'), 
                                         DecisionTreeClassifier(random_state=42, max_depth=10)),
            'Random Forest': make_pipeline(SimpleImputer(strategy='median'), 
                                         RandomForestClassifier(random_state=42, n_estimators=100))
        }
        
        results = {}
        print(f"\n{dataset_name} Performance:")
        print("-" * 80)
        print(f"{'Algorithm':<18} {'Accuracy':<10} {'Precision':<11} {'Recall':<9} {'F1-Score':<9}")
        print("-" * 80)
        
        for name, model in models.items():
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision, 
                'recall': recall,
                'f1': f1
            }
            
            print(f"{name:<18} {accuracy:<10.4f} {precision:<11.4f} {recall:<9.4f} {f1:<9.4f}")
        
        # Find best performing algorithm
        best_algo = max(results.keys(), key=lambda x: results[x]['f1'])
        print(f"\nBest Algorithm for {dataset_name}: {best_algo} (F1: {results[best_algo]['f1']:.4f})")
        
        return results, best_algo
    
    # Compare algorithms on both datasets
    approval_results, best_approval = compare_algorithms(X_approval, y_approval, "APPROVAL DATASET")
    payment_results, best_payment = compare_algorithms(X_payment, y_payment, "PAYMENT DATASET")
    
    # 4. DATASET COMPLEMENTARITY ANALYSIS
    print(f"\n=== 4. DATASET COMPLEMENTARITY ANALYSIS ===")
    
    # Compare common features
    approval_features = set(loan_approval.columns)
    payment_features = set(loan_payment.columns)
    
    common_features = approval_features.intersection(payment_features)
    unique_approval = approval_features - payment_features
    unique_payment = payment_features - approval_features
    
    print(f"Common Features ({len(common_features)}): {sorted(list(common_features))}")
    print(f"\nUnique to Approval Dataset ({len(unique_approval)}): {sorted(list(unique_approval))}")
    print(f"\nUnique to Payment Dataset ({len(unique_payment)}): {sorted(list(unique_payment))}")
    
    # Analyze correlation between similar features if available
    complementarity_score = len(common_features) / max(len(approval_features), len(payment_features))
    print(f"\nComplementarity Score: {complementarity_score:.3f}")
    print("Interpretation:")
    if complementarity_score > 0.7:
        print("- HIGH overlap: Datasets are similar and potentially redundant")
    elif complementarity_score > 0.3:
        print("- MEDIUM overlap: Datasets are complementary with shared foundation")
    else:
        print("- LOW overlap: Datasets capture different aspects of lending")
    
    # 5. BUSINESS RECOMMENDATIONS
    print(f"\n=== 5. BUSINESS RECOMMENDATIONS ===")
    
    print("Based on the analysis:")
    print("\n📊 FEATURE INSIGHTS:")
    if approval_targets and payment_targets:
        print(f"• Top approval factors: Focus on {', '.join(top_approval_factors['Feature'].head(3).tolist())}")
        print(f"• Top payment factors: Monitor {', '.join(top_payment_factors['Feature'].head(3).tolist())}")
    
    print(f"\n🤖 ALGORITHM RECOMMENDATIONS:")
    print(f"• For loan approval decisions: Use {best_approval}")
    print(f"• For payment default prediction: Use {best_payment}")
    
    print(f"\n💼 BUSINESS STRATEGY:")
    if complementarity_score > 0.5:
        print("• Datasets show good overlap - can build unified lending model")
        print("• Use approval model for initial screening")
        print("• Apply payment model for ongoing risk monitoring")
    else:
        print("• Datasets capture different lending aspects")
        print("• Use approval dataset for underwriting decisions")
        print("• Use payment dataset for portfolio risk management")
    
    print(f"\n🎯 OPERATIONAL RECOMMENDATIONS:")
    print("• Implement real-time scoring using top factors")
    print("• Set up automated alerts for high-risk profiles") 
    print("• Regular model retraining with new data")
    print("• A/B testing for model performance validation")
    
    return {
        'approval_results': approval_results,
        'payment_results': payment_results,
        'top_approval_factors': top_approval_factors,
        'top_payment_factors': top_payment_factors,
        'complementarity_score': complementarity_score
    }

if __name__ == "__main__":
    results = analyze_loan_datasets()