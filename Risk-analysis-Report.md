---

# **1 Data Science Role**

## **1.1 Job Description and Role Identification**

* **Analyze credit risk data**
  → Look at customer loan and credit card data to find out who are likely to miss payments.

* **Produce reports & recommendations**
  → Use visualization tools to create reports that clearly show what's going on, and suggest actions.

* **Monitor portfolio performance**
  → Keep track of how groups of customers are performing — are they paying back loans? Is risk increasing?

* **Run impact analysis**
  → If the company wants to change a rule or policy (like loan approval criteria), use data to predict the effects before it happens.

* **Support audits & governance**
  → Help prepare reports and insights needed for meetings or reviews.

**Job listing URL:** [https://builtin.com/job/credit-risk-analyst-pay/6796443](https://builtin.com/job/credit-risk-analyst-pay/6796443)
**Job listing cite:** [1]

---

## **1.2 Industry Context**

This is the credit risk/FinTech lending industry, which lends to individuals or entities and uses data to assess repayment likelihood. Companies analyze credit histories, and customer behavior to make lending decisions. Use data estimate default likelihood with ML models and predict the fully repaid likelihood.

---

## **1.3 Role Focus**

This role focuses on credit risk and lending in finance. Data scientists solve problems like predicting loan defaults, detecting fraud, and optimizing lending strategies. They help institutions reduce risk, and approve the right customers. Machine learning are often used in the process that help sustainable growth.

---

## **1.4 Value Creation**

This role helps the company avoid lending money to people or small business who might not pay it back. You have to analyze data and give smart advice or insights. You help the business grow safely, manage the loss, and make better decisions about loans and credit cards.

---

## **1.5 Values and Culture Alignment**

The company offers a flexible working environment. They advocate a safe, healthy, engaging, and productive working environment for all employees, whether that be in your home, the office or a combination of both.

---

## **1.6 Diversity and Inclusion**

People from different backgrounds notice different patterns from the data. This helps catch blind spots, write better code and build more reliable models. Also, when the team is diverse, they're more likely to notice unfair patterns—like if models have bias on certain groups they can fix them.

---

## **1.7 Cover Letter**

The cover letter is written as a stand-alone letter on the last page.

---

# **2 Data Set**

(1) **Loan Approval Dataset** (20,000 rows, 39 columns) [5]

- **Features:** Applicant's income, debts, and other details.
- **Target:** The target variable is _LoanApproved_ (1 = approved).
- **Approved:** 23.9% (imbalanced)

(2) **Loan Dataset** (9,578 rows, 17 columns) [6]

- **Features:** Interest rates, monthly loan payments, and other financial details.
- **Target:** Focus on repayment behavior. The target variable is _fully.paid_ (0 = not fully repaid).
- **Not Fully Paid:** 16.0% (imbalanced)

**Reason for choice:**
The approval dataset helps predict whether a loan should be approved, and the payment dataset helps predict whether an approved loan will be repaid in full. This combination helps loan company assess both approval eligibility and repayment likelihood after lending.

---

# **3 Data Analysis**

For this study, I used two datasets related to loan approval and repayment behaviour for credit risk prediction.
I approached the problem in two parts:

1. **Loan Approval Prediction** – predicting whether a customer should be approved for a loan.
2. **Loan Repayment Prediction** – predicting whether an approved loan will be fully repaid.

---

## **Machine Learning Models**

To gain insights, I applied:

1. **Logistic Regression** – a simple model that works well for binary classification problems. [4]
2. **Decision Tree** – find non-linear relationships and can provide better prediction outcomes in some cases. [2]

I choose these two ML models because they use different learning methods – an easy way and a more complex way – so their outcome comparison can be more meaningful.

---

## **Method**

- For each dataset, I look at the target distribution to find imbalance.
- Data was preprocessed (one-hot encoding, scaling, and median imputation for missing values).
- I trained and evaluated the models using 80–20 train-test split.
- Evaluated model performance using Precision, Recall, and F1-Score.
- Get key predictors using feature importance function. [3]

---

## **Evaluation Metrics**

Precision was prioritised because approving risky applicants (false positives) in loan approval or predicting repayment incorrectly can lead to significant financial risk.

---

## **Data Preparation Highlights**

- **Data cleaning:** Removed cheating features such as _RiskScore_ and _not.fully.paid_.
- **Feature engineering:** Used one-hot encoding for categorical variables and Standard Scaler to numeric features.
- **Data balancing (Approval):** Up-sampled approved cases from 23.9% to 50%.
- **Data balancing (Repayment):** Up-sampled defaults from 16% to 50%.

---

## **Findings**

### **• Loan_APorNAP2 (Loan Approval):**

- Logistic Regression achieved a precision of **93.84%**, while Decision Tree scored **91.18%**.
- **Key predictors:** Debt To Income Ratio, Monthly Income, Annual Income.
- **Insights:** Applicants with low debt-to-income ratios and high income levels were more likely to be approved.

### **• Loan_PAorNPA2 (Loan Repayment):**

- Decision Tree achieved a precision of **72.15%**, outperforming Logistic Regression at **61.40%**.
- **Key predictors:** Interest Rate, Annual Income, Monthly Debt Payments.
- **Insights:** Borrowers with lower interest rates, smaller monthly debt payments, and lower credit card utilisation were more likely to repay in full.

---

## **Business Insights**

- **Operational focus:**
  – _Approval:_ monitor applicants' debt-to-income ratio and income before approval.
  – _Repayment:_ pay attention on Interest rate, income for higher-risk customers.

- **Data handling:** Continue balancing data to prevent model bias as data expand.

- **Risk mitigation:** Higher precision reduces the likelihood of approving risky applicants and improves repayment reliability.

---

# **References**

[1] Built In. Credit Risk Analyst – Pay – Latitude Financial Services. [https://builtin.com/job/credit-risk-analyst-pay/6796443](https://builtin.com/job/credit-risk-analyst-pay/6796443).
[2] GeeksforGeeks. Decision Tree in Machine Learning – Introduction & Example. [https://www.geeksforgeeks.org/machine-learning/decision-tree-introduction-example/](https://www.geeksforgeeks.org/machine-learning/decision-tree-introduction-example/).
[3] GeeksforGeeks. Understanding Feature Importance and Visualization of Tree Models. [https://www.geeksforgeeks.org/machine-learning/understanding-feature-importance-and-visualization-of-tree-models/](https://www.geeksforgeeks.org/machine-learning/understanding-feature-importance-and-visualization-of-tree-models/).
[4] GeeksforGeeks. Understanding Logistic Regression. [https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/](https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/).
[5] Kaggle. Financial Risk for Loan Approval. [https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval/data](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval/data).
[6] Kaggle. Loan Data. [https://www.kaggle.com/datasets/itssuru/loan-data](https://www.kaggle.com/datasets/itssuru/loan-data).

---
