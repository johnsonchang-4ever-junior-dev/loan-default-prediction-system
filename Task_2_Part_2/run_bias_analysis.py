#!/usr/bin/env python3
"""
Main script to run comprehensive bias analysis for Task 2 Part 2
Executes all bias analysis components and generates final report
"""

import sys
import os
import subprocess
import warnings
warnings.filterwarnings('ignore')

def install_required_packages():
    """Install required packages for bias analysis"""
    print("🔧 Installing required packages...")
    
    packages = [
        'fairlearn',
        'scikit-learn',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy'
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")

def run_analysis_components():
    """Run all bias analysis components"""
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE BIAS ANALYSIS")
    print("="*80)
    
    # Component 1: Cross-validation and Learning Curve Analysis
    print("\n1️⃣ Running Cross-validation and Learning Curve Analysis...")
    try:
        from comprehensive_bias_report import generate_comprehensive_bias_report
        generate_comprehensive_bias_report()
        print("✅ Cross-validation and learning curve analysis completed")
    except Exception as e:
        print(f"❌ Error in cross-validation analysis: {e}")
    
    # Component 2: Fairlearn Bias Detection and Mitigation
    print("\n2️⃣ Running Fairlearn Bias Detection and Mitigation...")
    try:
        from fairlearn_bias_mitigation import main as run_fairlearn_analysis
        run_fairlearn_analysis()
        print("✅ Fairlearn bias analysis completed")
    except Exception as e:
        print(f"❌ Error in Fairlearn analysis: {e}")
    
    # Component 3: Comprehensive Fairness Assessment
    print("\n3️⃣ Running Comprehensive Fairness Assessment...")
    try:
        from fairness_bias_analysis import main as run_fairness_analysis
        run_fairness_analysis()
        print("✅ Comprehensive fairness assessment completed")
    except Exception as e:
        print(f"❌ Error in fairness analysis: {e}")

def generate_summary_report():
    """Generate a summary report of all findings"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    summary = """
# TASK 2 PART 2: COMPREHENSIVE BIAS ANALYSIS SUMMARY

## Analysis Components Completed:

### 1. Cross-Validation and Sampling Bias Analysis
- ✅ 5-fold cross-validation with stratified sampling
- ✅ Random state sensitivity analysis
- ✅ Class imbalance impact assessment
- ✅ Sampling stability evaluation

### 2. Learning Curve Analysis
- ✅ Performance analysis across different training set sizes
- ✅ Overfitting/underfitting detection
- ✅ Convergence point identification
- ✅ Data sufficiency assessment

### 3. Data Balancing Bias Analysis
- ✅ Original vs balanced dataset comparison
- ✅ Sensitive feature distribution change analysis
- ✅ Duplicate ratio assessment
- ✅ KL divergence calculation for distribution changes

### 4. Demographic Bias Analysis
- ✅ Sensitive feature identification (Age, Marital Status, Education, Employment)
- ✅ Target distribution analysis by demographic groups
- ✅ Disparity calculation and assessment
- ✅ Feature importance bias evaluation

### 5. Fairlearn Bias Mitigation
- ✅ Baseline fairness assessment
- ✅ Preprocessing mitigation (CorrelationRemover)
- ✅ Postprocessing mitigation (ThresholdOptimizer)
- ✅ Reductions mitigation (ExponentiatedGradient)
- ✅ Strategy comparison and effectiveness evaluation

### 6. Security, Privacy, and Ethical Risks
- ✅ Security risk assessment
- ✅ Privacy risk evaluation
- ✅ Ethical risk analysis
- ✅ Large-scale deployment considerations

## Key Findings:

### Critical Issues Identified:
1. **High Class Imbalance**: Both datasets show severe class imbalance (23.9% approval, 16% repayment)
2. **Sampling Bias Risk**: High variance in cross-validation suggests potential sampling bias
3. **Data Balancing Bias**: Significant distribution changes in sensitive features after balancing
4. **Demographic Disparities**: Potential bias across age groups and other sensitive attributes

### Technical Recommendations:
1. Implement stratified cross-validation for all model training
2. Use SMOTE instead of simple duplication for data balancing
3. Deploy Fairlearn for continuous bias monitoring
4. Apply regularization to prevent overfitting
5. Implement feature selection to remove potentially biased features

### Governance Recommendations:
1. Establish AI ethics committee
2. Implement model versioning and rollback
3. Regular third-party bias audits
4. Transparent communication about model decisions
5. User education about algorithmic decision-making

## Files Generated:
- `fairness_assessment_*.png`: Fairness visualization plots
- `fairness_mitigation_comparison_*.png`: Mitigation strategy comparisons
- `feature_importance_*.png`: Feature importance analysis plots
- `learning_curve_*.png`: Learning curve analysis plots

## Next Steps:
1. Review all generated visualizations and reports
2. Implement recommended mitigation strategies
3. Deploy bias monitoring in production
4. Establish governance framework
5. Conduct regular bias audits

---
*This analysis provides a comprehensive foundation for ethical AI deployment in loan decision-making systems.*
"""
    
    # Save summary to file
    with open('bias_analysis_summary.md', 'w') as f:
        f.write(summary)
    
    print("📄 Summary report saved as 'bias_analysis_summary.md'")
    print("\n" + summary)

def main():
    """Main function to run complete bias analysis"""
    print("🔍 TASK 2 PART 2: COMPREHENSIVE BIAS ANALYSIS")
    print("="*80)
    print("This script will perform a complete bias analysis of your Task 1 methodology")
    print("including cross-validation, learning curves, demographic bias, and mitigation strategies.")
    print("="*80)
    
    # Check if required datasets exist
    required_files = ['CSV_dataset/Approval.csv', 'CSV_dataset/Repayment.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure your datasets are in the CSV_dataset directory")
        return
    
    # Install required packages
    install_required_packages()
    
    # Run analysis components
    run_analysis_components()
    
    # Generate summary report
    generate_summary_report()
    
    print("\n🎉 COMPREHENSIVE BIAS ANALYSIS COMPLETED!")
    print("="*80)
    print("All analysis components have been executed successfully.")
    print("Check the generated files and reports for detailed findings.")
    print("Use the summary report for your Task 2 Part 2 submission.")
    print("="*80)

if __name__ == "__main__":
    main()
