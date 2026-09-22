# Customer Churn Prediction & Retention Intelligence

An end-to-end Data Analytics and Machine Learning project that analyzes customer behavior, predicts churn risk, estimates revenue at risk, and provides retention-focused insights through an interactive Streamlit dashboard.

## 🚀 Live Demo

https://customer-churn-retention-intelligence-by-aditi.streamlit.app/

## 📌 Project Overview

Customer churn is an important business problem because losing customers can directly affect revenue.

This project analyzes customer data to:
- Understand customer churn patterns
- Identify important churn-related factors
- Measure customer health
- Segment customers based on behavior and value
- Predict customer churn probability
- Estimate monthly revenue at risk
- Prioritize customers for retention
- Recommend suitable retention actions

## 📊 Key Results

| Metric | Result |
|---|---:|
| Total Customers | 10,000 |
| Overall Churn Rate | 10.21% |
| Customers At Risk | 49.72% |
| Monthly Revenue at Risk | ₹35,662.83 |
| High Priority Customers | 815 |
| Medium Priority Customers | 724 |

## 🔍 Project Workflow

```text
Data → EDA → Churn Driver Analysis → Customer Health Score
     → Customer Segmentation → Churn Prediction
     → Revenue at Risk → Retention Priority
     → Retention Recommendations → Streamlit Dashboard
```

## 📈 Churn Driver Analysis

The project analyzes customer behavior and identifies variables associated with different churn rates.

Important churn-related factors observed in the analysis include:
- Monthly logins
- Tenure
- CSAT score
- Payment failures
- Last login activity
- Customer segment

## ❤️ Customer Health Score

A business-defined health score is created using:
- Monthly logins
- Tenure
- CSAT score
- Payment failures
- Last login activity

Customers are categorized as:
- **Critical**
- **At Risk**
- **Healthy**

## 👥 Customer Segmentation

K-Means clustering is used to group customers based on behavior, engagement, experience, and value.

Two groups were identified:
1. **Lower-Value / Newer Customers**
2. **High-Value / Long-Term Customers**

StandardScaler is used before K-Means because clustering is distance-based. PCA is used to visualize the clusters.

## 🤖 Churn Prediction

Three classification models were evaluated:
- Logistic Regression
- Random Forest
- Gradient Boosting

Gradient Boosting provided the strongest baseline performance and was selected for further tuning.

GridSearchCV with 5-fold cross-validation was used for hyperparameter tuning.

The tuned model achieved a test ROC-AUC of approximately **0.809**.

A churn probability threshold of **0.20** was used for the retention-focused prediction because it provided higher recall and F1 among the tested thresholds.

## 💰 Revenue at Risk

Revenue at Risk is estimated using:

```text
Revenue at Risk = Monthly Fee × Churn Probability
```

Estimated monthly revenue at risk:

**₹35,662.83**

This is an estimated revenue exposure based on predicted churn probability, not actual revenue already lost.

## 🎯 Retention Priority

Customers are prioritized using churn probability and customer value.

- **High:** churn probability ≥ 0.20 and monthly fee at or above the median
- **Medium:** churn probability ≥ 0.20 and monthly fee below the median
- **Low:** remaining customers

## 💡 Retention Recommendations

Rule-based recommendations are generated from customer risk factors:
- **Payment support** → customers with payment failures
- **Customer support follow-up** → customers with lower CSAT
- **Re-engagement campaign** → customers with lower monthly logins
- **Regular engagement** → remaining customers

## 🖥️ Streamlit Dashboard

The dashboard includes:
- Overview
- Visualization
- Customer Segmentation
- Customer Health
- Churn Prediction
- Revenue at Risk
- Retention Priority
- Recommendations
- Customer Search
- About

## 🛠️ Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Plotly**
- **Streamlit**
- **Streamlit Option Menu**
- **Joblib**
- **GitHub**
- **Streamlit Community Cloud**

## 📁 Project Structure

```text
customer-churn-retention-intelligence/
│
├── app/
│   └── app.py
├── Data/
│   └── customer_churn_business_dataset.csv
├── Models/
│   └── churn_model.pkl
├── notebook/
│   └── customer_churn_analysis.ipynb
├── .streamlit/
│   └── config.toml
├── .gitignore
├── requirements.txt
└── README.md
```

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/Aditi-kumawat14/customer-churn-retention-intelligence.git
```

Go to the project folder:

```bash
cd customer-churn-retention-intelligence
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/app.py
```

Local URL:

```text
http://localhost:8501
```

## 📌 Key Business Insight

This project combines descriptive analytics, customer health scoring, machine learning, revenue exposure analysis, customer segmentation, and retention recommendations into one dashboard.

It moves from:

**"Which customers are likely to churn?"**

to:

**"Which customers should receive retention attention and why?"**

## 👩‍💻 Author

**Aditi Kumawat**

Computer Engineering | Data Analytics | Machine Learning | Python

GitHub: https://github.com/Aditi-kumawat14

Live Dashboard: https://customer-churn-retention-intelligence-by-aditi.streamlit.app/
