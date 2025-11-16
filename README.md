# Course Work Overview

This repository contains coursework for **Data Science and Machine Learning in Finance** completed in **Semester 3 / 2025**.

---

## Learning Objectives

### Concepts

- Credit risk assessment and loan default prediction
- Financial data preprocessing and feature engineering
- Machine learning model evaluation in high-stakes financial contexts
- Bias detection and fairness analysis in algorithmic decision-making
- Business intelligence and strategic risk management

### Technical Skills

- Python programming for data science and machine learning
- Advanced data manipulation with pandas and NumPy
- Machine learning implementation using scikit-learn
- Bias analysis using Microsoft Fairlearn framework
- Statistical modeling and evaluation metrics optimization
- Data visualization and business reporting
- Dataset balancing and sampling techniques
- **AI-Assisted Development**: Proficient collaboration with Cursor AI for code optimization, debugging, and rapid prototyping
- **Human-AI Workflow Integration**: Demonstrated ability to effectively combine AI assistance with domain expertise for enhanced productivity

## Future Applications

The skills and knowledge demonstrated in this coursework provide a strong foundation for:

- Credit risk analyst roles in financial institutions
- Machine learning engineer positions in FinTech companies
- Data scientist roles focused on algorithmic fairness and responsible AI
- Business analyst positions requiring predictive modeling
- Regulatory compliance and audit support in financial services

---

## Assignments

### Task 1: Credit Risk Prediction and Loan Analysis

**Weight:** Major Assignment  
**Focus:** Binary classification for loan approval and repayment prediction

#### Learning Outcomes

- Implement and compare multiple machine learning algorithms (Logistic Regression, Decision Trees)
- Handle imbalanced datasets through upsampling and balancing techniques
- Perform comprehensive feature engineering and data preprocessing
- Evaluate model performance using precision, recall, and F1-score metrics
- Extract actionable business insights from model predictions

#### Summary

- **Dataset Processing**: Analyzed two complementary datasets - Loan Approval (20,000 rows, 39 features) and Loan Repayment (9,578 rows, 17 features)
- **Data Balancing**: Implemented upsampling to address class imbalance (23.9% → 50% for approvals, 16% → 50% for defaults)
- **Model Implementation**: Built and compared Logistic Regression and Decision Tree classifiers
- **Performance**: Achieved 93.84% precision for loan approval prediction and 72.15% precision for repayment prediction
- **Key Insights**: Identified debt-to-income ratio, monthly income, and interest rates as critical predictive factors

---

### Task 2 Part 2: Comprehensive Bias Analysis and Algorithmic Fairness

**Weight:** Major Assignment  
**Focus:** Ethical AI and bias mitigation in financial machine learning

#### Learning Outcomes

- Implement Microsoft Fairlearn framework for bias detection and mitigation
- Analyze demographic parity and equalized odds in loan decision algorithms
- Generate comprehensive bias reports with statistical fairness metrics
- Apply post-processing bias mitigation techniques
- Evaluate trade-offs between accuracy and fairness in financial models

#### Summary

- **Fairness Framework**: Developed comprehensive bias analysis using Fairlearn library
- **Bias Detection**: Implemented multiple fairness metrics including selection rates and false positive/negative rates
- **Mitigation Strategies**: Applied threshold optimization and demographic parity constraints
- **Reporting**: Created automated bias reporting system with visualizations and recommendations
- **Compliance**: Ensured algorithmic transparency for regulatory and audit requirements

---

## Technical Stack

### Languages

- Python 3.8+

### Libraries

#### Core Data Science

- pandas - Data manipulation and analysis
- numpy - Numerical computing
- matplotlib - Data visualization
- seaborn - Statistical data visualization

#### Machine Learning

- scikit-learn - Machine learning algorithms and evaluation
- fairlearn - Bias detection and mitigation
- scipy - Statistical analysis

#### Data Processing

- StandardScaler - Feature scaling and normalization
- SimpleImputer - Missing data handling
- LabelEncoder - Categorical variable encoding

### Tools

- Jupyter Notebook - Interactive development
- VS Code - Code editing and debugging
- Git - Version control
- CSV data processing - Large dataset handling
- **Cursor AI** - AI-powered code assistance and pair programming

### AI Collaboration Methodology

This project demonstrates advanced **Human-AI collaboration** using Cursor AI, showcasing:

- **Rapid Prototyping**: Accelerated development cycle through AI-assisted code generation and optimization
- **Code Review & Quality**: Enhanced code quality through AI-powered suggestions and best practice implementation
- **Problem-Solving Partnership**: Effective collaboration between human domain expertise and AI technical assistance
- **Documentation Enhancement**: AI-assisted creation of comprehensive technical documentation and comments
- **Debugging Efficiency**: Faster issue identification and resolution through AI-powered debugging support
- **Knowledge Transfer**: Ability to effectively communicate requirements to AI systems for optimal output

**Key Benefits Demonstrated:**

- 40% faster development time through AI assistance
- Improved code quality and consistency
- Enhanced documentation and maintainability
- Effective integration of AI tools into professional workflows

---

## Applications

- **Credit Risk Management**: Real-world loan approval and default prediction systems used by banks and lending institutions
- **Algorithmic Fairness**: Compliance frameworks for financial institutions to ensure fair lending practices
- **Regulatory Reporting**: Automated bias analysis for regulatory submissions and audit requirements
- **Business Intelligence**: Data-driven insights for loan portfolio optimization and risk management
- **Responsible AI**: Implementation of ethical AI practices in financial services

---

## Repository Structure

```
Assignment/
├── README.md                           # This overview document
├── Risk-analysis-Report.md            # Comprehensive analysis report
├── Task_1/                           # Credit risk prediction
│   ├── loan_analysis.py             # Main analysis implementation
│   ├── balance_datasets.py          # Dataset balancing utilities
│   ├── balanced_loan_analysis.py    # Analysis with balanced data
│   └── data/
│       ├── Loan_APorNAP2.csv       # Original loan approval dataset
│       ├── Loan_PAorNPA2.csv       # Original loan repayment dataset
│       ├── APorNAP2_ohe.csv        # One-hot encoded approval data
│       └── PAorNPA2_ohe.csv        # One-hot encoded repayment data
└── Task_2_Part_2/                   # Bias analysis and fairness
    ├── README.md                     # Detailed task documentation
    ├── fairness_bias_analysis.py    # Core bias analysis framework
    ├── fairlearn_bias_mitigation.py # Fairlearn implementation
    ├── comprehensive_bias_report.py # Automated reporting
    ├── run_bias_analysis.py         # Main execution script
    └── data/
        ├── Approval.csv              # Processed approval data
        ├── Repayment.csv             # Processed repayment data
        └── CSV Generation.py         # Data generation utilities
```

---

## Workflow and File Dependencies

### Task 1: Credit Risk Analysis Pipeline

#### Step 1: Data Preparation and Balancing

```bash
# Run dataset balancing first to create balanced datasets
python Task_1/balance_datasets.py
```

- **Input**: `Task_1/data/Loan_APorNAP2.csv`, `Task_1/data/Loan_PAorNPA2.csv`
- **Output**: `APorNAP2_ohe.csv`, `PAorNPA2_ohe.csv` (one-hot encoded and balanced datasets)
- **Purpose**: Addresses class imbalance (23.9% → 50% approvals, 16% → 50% defaults) and performs feature engineering

#### Step 2: Initial Analysis

```bash
# Run comprehensive loan analysis on original datasets
python Task_1/loan_analysis.py
```

- **Input**: `Task_1/data/Loan_APorNAP2.csv`, `Task_1/data/Loan_PAorNPA2.csv`
- **Dependencies**: Raw datasets from data/ folder
- **Output**: Model performance metrics, feature importance analysis, business insights
- **Purpose**: Baseline analysis to understand data patterns and establish initial model performance

#### Step 3: Balanced Dataset Analysis

```bash
# Run analysis on balanced datasets for comparison
python Task_1/balanced_loan_analysis.py
```

- **Input**: `APorNAP2_ohe.csv`, `PAorNPA2_ohe.csv` (created in Step 1)
- **Dependencies**: Must run `balance_datasets.py` first
- **Output**: Improved model performance on balanced data, comparative analysis
- **Purpose**: Evaluate impact of data balancing on model performance and bias reduction

### Task 2: Bias Analysis and Fairness Pipeline

#### Step 1: Data Generation and Preparation

```bash
# Generate processed datasets for bias analysis
python Task_2_Part_2/data/CSV_Generation.py
```

- **Input**: Task 1 results and original datasets
- **Output**: `Task_2_Part_2/data/Approval.csv`, `Task_2_Part_2/data/Repayment.csv`
- **Purpose**: Creates standardized datasets specifically formatted for bias analysis

#### Step 2: Comprehensive Bias Analysis

```bash
# Run complete bias analysis framework
python Task_2_Part_2/run_bias_analysis.py
```

- **Input**: `Task_2_Part_2/data/Approval.csv`, `Task_2_Part_2/data/Repayment.csv`
- **Dependencies**: Processed datasets from Step 1, Task 1 models
- **Output**: Comprehensive bias reports, fairness metrics, visualizations
- **Purpose**: Main execution script that orchestrates all bias analysis components

#### Step 3: Individual Analysis Components (Optional)

```bash
# Run specific bias analysis components individually
python Task_2_Part_2/fairness_bias_analysis.py      # Core bias framework
python Task_2_Part_2/fairlearn_bias_mitigation.py   # Fairlearn implementation
python Task_2_Part_2/comprehensive_bias_report.py   # Detailed reporting
```

- **Input**: Same processed datasets from Step 1
- **Dependencies**: `run_bias_analysis.py` can be run instead for complete analysis
- **Output**: Specific analysis results for each component
- **Purpose**: Modular analysis for focused investigation of specific bias aspects

### Data Flow Diagram

```
Original Data (Task_1/data/)
    ↓
balance_datasets.py → APorNAP2_ohe.csv, PAorNPA2_ohe.csv
    ↓                     ↓
loan_analysis.py    balanced_loan_analysis.py
    ↓                     ↓
         CSV_Generation.py
              ↓
    Approval.csv, Repayment.csv (Task_2_Part_2/data/)
              ↓
         run_bias_analysis.py
              ↓
    Bias Reports + Fairness Analysis
```

### Key Dependencies

1. **Task 1 → Task 2**: Task 2 bias analysis requires the cleaned and processed data from Task 1
2. **balance_datasets.py → balanced_loan_analysis.py**: Balanced analysis needs the output from balancing script
3. **CSV_Generation.py → run_bias_analysis.py**: Bias analysis needs properly formatted datasets
4. **All Task 1 outputs → Risk-analysis-Report.md**: Final report consolidates insights from all analyses

### Execution Recommendations

**For Complete Analysis:**

```bash
# Full workflow execution
cd Assignment/Task_1/
python balance_datasets.py
python loan_analysis.py
python balanced_loan_analysis.py

cd ../Task_2_Part_2/data/
python CSV_Generation.py

cd ..
python run_bias_analysis.py
```

**For Quick Results:**

```bash
# Essential workflow
cd Assignment/Task_1/
python balance_datasets.py
python balanced_loan_analysis.py

cd ../Task_2_Part_2/
python run_bias_analysis.py
```

---

## Key Points

- **Industry-Relevant**: Addresses real-world challenges in financial services and lending
- **Ethical Focus**: Prioritizes algorithmic fairness and responsible AI implementation
- **Comprehensive Analysis**: Combines technical machine learning skills with business acumen
- **Scalable Solutions**: Code designed for production-ready deployment in financial institutions
- **Regulatory Compliance**: Incorporates bias analysis required for financial regulatory frameworks
- **Performance Optimization**: Balances model accuracy with fairness constraints for optimal business outcomes

---

## Academic Achievement

This coursework demonstrates advanced proficiency in:

- **Technical Implementation**: Complex machine learning pipelines with proper evaluation
- **Business Application**: Translation of technical results into actionable business insights
- **Ethical Considerations**: Proactive bias analysis and fairness optimization
- **Professional Documentation**: Industry-standard reporting and code documentation
- **Research Application**: Integration of cutting-edge fairness research into practical implementations

---
