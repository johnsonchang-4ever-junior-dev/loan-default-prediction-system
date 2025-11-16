# Balanced Loan Analysis: Balance datasets first, then evaluate
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def balanced_loan_analysis():
    """Balance datasets first by duplicating minority class, then evaluate"""
    
    print("=== BALANCED LOAN ANALYSIS (BALANCE DATA FIRST) ===\n")
    
    # 1. LOAD DATASETS
    print("1. LOADING OHE DATASETS...")
    try:
        approval_ohe = pd.read_csv('APorNAP2_ohe.csv')
        payment_ohe = pd.read_csv('PAorNPA2_ohe.csv')
        print(f"✓ Loaded APorNAP2_ohe.csv: {approval_ohe.shape}")
        print(f"✓ Loaded PAorNPA2_ohe.csv: {payment_ohe.shape}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    # 2. EXAMINE ORIGINAL DISTRIBUTIONS
    print(f"\n=== 2. ORIGINAL TARGET DISTRIBUTIONS ===")
    
    # Approval dataset
    if 'LoanApproved' in approval_ohe.columns:
        print(f"APPROVAL - LoanApproved distribution:")
        approval_dist = approval_ohe['LoanApproved'].value_counts().sort_index()
        print(approval_dist)
        print(f"Balance ratio: {approval_dist.min()}/{approval_dist.max()} = {approval_dist.min()/approval_dist.max():.3f}")
    
    # Payment dataset  
    if 'fully.paid' in payment_ohe.columns:
        print(f"\nPAYMENT - fully.paid distribution:")
        payment_dist = payment_ohe['fully.paid'].value_counts().sort_index()
        print(payment_dist)
        print(f"Balance ratio: {payment_dist.min()}/{payment_dist.max()} = {payment_dist.min()/payment_dist.max():.3f}")
    else:
        print("Warning: fully.paid not found in payment dataset")
        print(f"Available columns: {list(payment_ohe.columns)}")
        return
    
    # 3. BALANCE DATASETS BY DUPLICATING MINORITY CLASS
    print(f"\n=== 3. BALANCING DATASETS ===")
    
    def balance_dataset(df, target_col, dataset_name):
        """Balance dataset by duplicating minority class rows"""
        print(f"\n{dataset_name} - Balancing {target_col}:")
        
        # Get value counts
        value_counts = df[target_col].value_counts()
        majority_class = value_counts.idxmax()
        minority_class = value_counts.idxmin()
        
        majority_count = value_counts[majority_class]
        minority_count = value_counts[minority_class]
        
        print(f"Majority class ({majority_class}): {majority_count}")
        print(f"Minority class ({minority_class}): {minority_count}")
        
        # Calculate how many times to duplicate minority class
        duplication_factor = majority_count // minority_count
        remainder = majority_count % minority_count
        
        print(f"Need to duplicate minority class {duplication_factor} times + {remainder} extra rows")
        
        # Separate majority and minority classes
        majority_df = df[df[target_col] == majority_class].copy()
        minority_df = df[df[target_col] == minority_class].copy()
        
        # Duplicate minority class
        duplicated_minority = []
        
        # Full duplications
        for _ in range(duplication_factor):
            duplicated_minority.append(minority_df.copy())
        
        # Partial duplication for remainder
        if remainder > 0:
            partial_minority = minority_df.sample(n=remainder, random_state=42).copy()
            duplicated_minority.append(partial_minority)
        
        # Combine all duplicated minority data
        if duplicated_minority:
            all_minority = pd.concat(duplicated_minority, ignore_index=True)
        else:
            all_minority = minority_df.copy()
        
        # Combine with majority class
        balanced_df = pd.concat([majority_df, all_minority], ignore_index=True)
        
        # Shuffle the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Original shape: {df.shape}")
        print(f"Balanced shape: {balanced_df.shape}")
        print(f"New distribution:")
        print(balanced_df[target_col].value_counts().sort_index())
        
        return balanced_df
    
    # Balance both datasets
    approval_balanced = balance_dataset(approval_ohe, 'LoanApproved', 'APPROVAL DATASET')
    payment_balanced = balance_dataset(payment_ohe, 'fully.paid', 'PAYMENT DATASET')
    
    # 4. EXCLUDE RISKSCORE FROM BALANCED DATASETS
    print(f"\n=== 4. EXCLUDE RISKSCORE FROM FEATURES ===")
    
    def remove_riskscore(df, dataset_name):
        """Remove RiskScore columns"""
        riskscore_cols = [col for col in df.columns if 'risk' in col.lower() and 'score' in col.lower()]
        if riskscore_cols:
            df = df.drop(columns=riskscore_cols)
            print(f"{dataset_name}: Removed {riskscore_cols}")
        return df
    
    approval_clean = remove_riskscore(approval_balanced, 'APPROVAL')
    payment_clean = remove_riskscore(payment_balanced, 'PAYMENT')
    
    # 5. FEATURE IMPORTANCE ON BALANCED DATA
    print(f"\n=== 5. TOP FACTORS (BALANCED DATA, NO RISKSCORE) ===")
    
    def get_feature_importance_balanced(df, target_col, dataset_name):
        """Get feature importance from balanced dataset"""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical columns
        for col in X.select_dtypes(include=['object']).columns:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Train Random Forest on balanced data
        rf = RandomForestClassifier(n_estimators=100, random_state=42)  # No class_weight needed - data is balanced
        rf.fit(X_imputed, y)
        
        # Get importance
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\n{dataset_name} - Top 10 Features (Balanced Data):")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df.head(3)
    
    top_approval_balanced = get_feature_importance_balanced(approval_clean, 'LoanApproved', 'APPROVAL BALANCED')
    top_payment_balanced = get_feature_importance_balanced(payment_clean, 'fully.paid', 'PAYMENT BALANCED')
    
    # 6. ALGORITHM EVALUATION ON BALANCED DATA
    print(f"\n=== 6. ALGORITHM EVALUATION (BALANCED DATA, PRECISION-FOCUSED) ===")
    
    def evaluate_algorithms_balanced(df, target_col, dataset_name):
        """Evaluate algorithms on balanced dataset"""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical columns
        for col in X.select_dtypes(include=['object']).columns:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        
        # Train-test split (stratification should work well on balanced data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Define models (no class_weight needed - data is balanced)
        models = {
            'Logistic Regression': make_pipeline(
                SimpleImputer(strategy='median'), 
                StandardScaler(), 
                LogisticRegression(random_state=42, max_iter=1000)
            ),
            'Decision Tree': make_pipeline(
                SimpleImputer(strategy='median'), 
                DecisionTreeClassifier(random_state=42, max_depth=10)
            ),
            'Random Forest': make_pipeline(
                SimpleImputer(strategy='median'), 
                RandomForestClassifier(random_state=42, n_estimators=100)
            )
        }
        
        results = {}
        print(f"\n{dataset_name} Performance (Balanced Data, Precision-Focused):")
        print("-" * 85)
        print(f"{'Algorithm':<18} {'Accuracy':<10} {'Precision':<11} {'Recall':<9} {'F1-Score':<9} {'Specificity':<11}")
        print("-" * 85)
        
        for name, model in models.items():
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # For binary classification
            if y.nunique() == 2:
                # Use the minority class (1) as positive class for precision
                precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
                recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
                f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
                
                # Calculate specificity (True Negative Rate)
                cm = confusion_matrix(y_test, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                else:
                    specificity = 0
            else:
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                specificity = 0
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision, 
                'recall': recall,
                'f1': f1,
                'specificity': specificity
            }
            
            print(f"{name:<18} {accuracy:<10.4f} {precision:<11.4f} {recall:<9.4f} {f1:<9.4f} {specificity:<11.4f}")
        
        # Best by precision
        best_algo = max(results.keys(), key=lambda x: results[x]['precision'])
        print(f"\nBest Algorithm for {dataset_name} (by Precision): {best_algo} (Precision: {results[best_algo]['precision']:.4f})")
        
        # Show confusion matrix
        best_model = models[best_algo]
        y_pred_best = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_best)
        print(f"\nConfusion Matrix for {best_algo}:")
        print(cm)
        print(f"Test set distribution: {pd.Series(y_test).value_counts().sort_index().to_dict()}")
        
        return results, best_algo
    
    # Evaluate both balanced datasets
    print(f"\n=== 6A. BALANCED APPROVAL DATASET EVALUATION ===")
    approval_results_balanced = evaluate_algorithms_balanced(approval_clean, 'LoanApproved', 'APPROVAL BALANCED')
    
    print(f"\n=== 6B. BALANCED PAYMENT DATASET EVALUATION ===")
    payment_results_balanced = evaluate_algorithms_balanced(payment_clean, 'fully.paid', 'PAYMENT BALANCED')
    
    # 7. COMPARISON WITH PREVIOUS UNBALANCED RESULTS
    print(f"\n=== 7. BALANCED VS UNBALANCED COMPARISON ===")
    
    print(f"\n📊 DATASET TRANSFORMATION SUMMARY:")
    print(f"• Approval: {approval_ohe.shape} → {approval_clean.shape}")
    print(f"• Payment: {payment_ohe.shape} → {payment_clean.shape}")
    
    print(f"\n🔍 TOP FEATURES COMPARISON (Balanced Data):")
    print(f"Approval Top 3: {', '.join(top_approval_balanced['Feature'].tolist())}")
    print(f"Payment Top 3: {', '.join(top_payment_balanced['Feature'].tolist())}")
    
    # Check if top features are shared
    approval_top = set(top_approval_balanced['Feature'].tolist())
    payment_top = set(top_payment_balanced['Feature'].tolist())
    shared_top = approval_top.intersection(payment_top)
    print(f"Shared top features: {shared_top if shared_top else 'None'}")
    
    print(f"\n🤖 BEST ALGORITHMS (Balanced Data):")
    print(f"Best for Approval: {approval_results_balanced[1]} (Precision: {approval_results_balanced[0][approval_results_balanced[1]]['precision']:.4f})")
    print(f"Best for Payment: {payment_results_balanced[1]} (Precision: {payment_results_balanced[0][payment_results_balanced[1]]['precision']:.4f})")
    
    # 8. BUSINESS RECOMMENDATIONS FOR BALANCED DATA
    print(f"\n=== 8. BUSINESS RECOMMENDATIONS (BALANCED DATA) ===")
    
    print(f"\n🎯 KEY FINDINGS FROM BALANCED DATA:")
    print(f"• Balanced approval model focuses on: {', '.join(top_approval_balanced['Feature'].tolist())}")
    print(f"• Balanced payment model focuses on: {', '.join(top_payment_balanced['Feature'].tolist())}")
    
    print(f"\n⚖️ BALANCING IMPACT:")
    print(f"• Approval precision: {approval_results_balanced[0][approval_results_balanced[1]]['precision']:.1%}")
    print(f"• Payment precision: {payment_results_balanced[0][payment_results_balanced[1]]['precision']:.1%}")
    print(f"• Balanced data provides more reliable feature importance")
    print(f"• Better representation of minority class patterns")
    
    print(f"\n💡 IMPLEMENTATION STRATEGY (BALANCED APPROACH):")
    print(f"• Use {approval_results_balanced[1]} for balanced approval decisions")
    print(f"• Use {payment_results_balanced[1]} for balanced payment prediction")
    print(f"• Data balancing reveals true feature importance")
    print(f"• Consider synthetic data generation for production")
    
    if shared_top:
        print(f"• Shared features {shared_top} are consistently important across both domains")
    
    return {
        'approval_balanced': approval_clean,
        'payment_balanced': payment_clean,
        'approval_results': approval_results_balanced,
        'payment_results': payment_results_balanced,
        'top_approval_factors': top_approval_balanced,
        'top_payment_factors': top_payment_balanced
    }

if __name__ == "__main__":
    results = balanced_loan_analysis()