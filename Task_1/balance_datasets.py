#!/usr/bin/env python3
"""
Dataset Balancing Script
Balances both loan datasets to 50-50 class distribution and exports them
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample
import os

def balance_dataset(df, target_col, random_state=42):
    """
    Balance dataset to 50-50 class distribution using upsampling
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        random_state: Random seed for reproducibility
    
    Returns:
        Balanced DataFrame with 50-50 class distribution
    """
    # Get class counts
    class_counts = df[target_col].value_counts()
    print(f"Original class distribution: {class_counts.to_dict()}")
    
    # Find minority and majority classes
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    minority_count = class_counts.min()
    majority_count = class_counts.max()
    
    print(f"Minority class: {minority_class} ({minority_count} samples)")
    print(f"Majority class: {majority_class} ({majority_count} samples)")
    
    # Separate majority and minority classes
    df_majority = df[df[target_col] == majority_class]
    df_minority = df[df[target_col] == minority_class]
    
    # Upsample minority class to match majority class
    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=majority_count,
        random_state=random_state
    )
    
    # Combine majority and upsampled minority
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Verify balance
    final_counts = df_balanced[target_col].value_counts()
    print(f"Balanced class distribution: {final_counts.to_dict()}")
    print(f"Total samples: {len(df_balanced)}")
    
    return df_balanced

def main():
    """Main function to balance both datasets"""
    print("=== BALANCING DATASETS TO 50-50 CLASS DISTRIBUTION ===")
    
    # Balance Loan_APorNAP2_ohe dataset (Loan Approval)
    print("\n--- Balancing Loan_APorNAP2_ohe (Loan Approval) ---")
    try:
        # Load the dataset
        print("Loading APorNAP2_ohe.csv...")
        Loan_APorNAP2_ohe = pd.read_csv('APorNAP2_ohe.csv')
        
        # Balance the dataset
        Loan_APorNAP2_ohe_balanced = balance_dataset(
            Loan_APorNAP2_ohe, 
            'LoanApproved'
        )
        
        # Export balanced dataset
        output_filename_1 = 'Loan_APorNAP2_ohe_balanced_50_50.csv'
        Loan_APorNAP2_ohe_balanced.to_csv(output_filename_1, index=False)
        print(f"✅ Exported balanced dataset to: {output_filename_1}")
        
    except Exception as e:
        print(f"❌ Error balancing Loan_APorNAP2_ohe: {e}")
        return
    
    # Balance Loan_PAorNPA2_ohe dataset (Fully Paid)
    print("\n--- Balancing Loan_PAorNPA2_ohe (Fully Paid) ---")
    try:
        # Load the dataset
        print("Loading PAorNPA2_ohe.csv...")
        Loan_PAorNPA2_ohe = pd.read_csv('PAorNPA2_ohe.csv')
        
        # Ensure we have the target column
        if 'fully.paid' not in Loan_PAorNPA2_ohe.columns:
            # Try to derive it
            if 'not.fully.paid' in Loan_PAorNPA2_ohe.columns:
                Loan_PAorNPA2_ohe['fully.paid'] = 1 - pd.to_numeric(Loan_PAorNPA2_ohe['not.fully.paid'], errors='coerce')
            elif 'not_fully_paid' in Loan_PAorNPA2_ohe.columns:
                Loan_PAorNPA2_ohe['fully.paid'] = 1 - pd.to_numeric(Loan_PAorNPA2_ohe['not_fully_paid'], errors='coerce')
            else:
                raise ValueError("Could not find or derive 'fully.paid' target column")
        
        # Clean target column
        Loan_PAorNPA2_ohe['fully.paid'] = pd.to_numeric(Loan_PAorNPA2_ohe['fully.paid'], errors='coerce')
        Loan_PAorNPA2_ohe = Loan_PAorNPA2_ohe[~Loan_PAorNPA2_ohe['fully.paid'].isna()].copy()
        Loan_PAorNPA2_ohe['fully.paid'] = Loan_PAorNPA2_ohe['fully.paid'].round().astype(int)
        
        # Balance the dataset
        Loan_PAorNPA2_ohe_balanced = balance_dataset(
            Loan_PAorNPA2_ohe, 
            'fully.paid'
        )
        
        # Export balanced dataset
        output_filename_2 = 'Loan_PAorNPA2_ohe_balanced_50_50.csv'
        Loan_PAorNPA2_ohe_balanced.to_csv(output_filename_2, index=False)
        print(f"✅ Exported balanced dataset to: {output_filename_2}")
        
    except Exception as e:
        print(f"❌ Error balancing Loan_PAorNPA2_ohe: {e}")
        return
    
    # Summary of balanced datasets
    print("\n=== SUMMARY ===")
    print("Balanced datasets created with 50-50 class distribution:")
    print(f"1. {output_filename_1} - Balanced loan approval dataset")
    print(f"2. {output_filename_2} - Balanced fully paid dataset")
    
    # Verify file sizes
    if os.path.exists(output_filename_1):
        size_1 = os.path.getsize(output_filename_1) / (1024*1024)  # MB
        print(f"   - {output_filename_1}: {size_1:.2f} MB")
    if os.path.exists(output_filename_2):
        size_2 = os.path.getsize(output_filename_2) / (1024*1024)  # MB
        print(f"   - {output_filename_2}: {size_2:.2f} MB")
    
    print("\n✅ Dataset balancing and export completed!")
    print("You can now use these balanced datasets for training more robust ML models.")
    
    # Show sample statistics
    print("\n=== SAMPLE STATISTICS ===")
    try:
        print(f"Original Loan_APorNAP2_ohe shape: {Loan_APorNAP2_ohe.shape}")
        print(f"Balanced Loan_APorNAP2_ohe shape: {Loan_APorNAP2_ohe_balanced.shape}")
        print(f"Original Loan_PAorNPA2_ohe shape: {Loan_PAorNPA2_ohe.shape}")
        print(f"Balanced Loan_PAorNPA2_ohe shape: {Loan_PAorNPA2_ohe_balanced.shape}")
    except:
        pass

if __name__ == "__main__":
    main()
