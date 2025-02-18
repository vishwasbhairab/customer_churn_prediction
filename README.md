# Customer Churn Prediction using XGBoost 🚀

## 📌 Project Overview

Customer churn prediction is crucial for businesses to retain customers and improve their services. This project builds a **machine learning model** to predict whether a customer will churn or not, using the **IBM Telco Customer Churn Dataset**. The model leverages **XGBoost** for high accuracy and explainability.

## 📂 Dataset

- **Dataset Name**: IBM Telco Customer Churn
- **Source**: [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Features**:
  - Customer demographics (gender, senior citizen, tenure, etc.)
  - Service subscriptions (phone service, internet service, contract type, etc.)
  - Billing details (monthly charges, total charges, payment method, etc.)
  - **Target Variable**: `Churn` (Yes/No)

## 🛠️ Tech Stack

- **Programming Language**: Python 🐍
- **Libraries Used**:
  - `pandas`, `numpy` (Data Processing)
  - `scikit-learn` (Preprocessing & ML Models)
  - `imblearn` (Handling Imbalanced Data)
  - `XGBoost` (Machine Learning Model)
  - `matplotlib`, `seaborn` (Data Visualization)

## 🔍 Methodology

1. **Data Preprocessing**:

   - Handling missing values
   - Encoding categorical features (One-Hot Encoding)
   - Feature scaling using `MinMaxScaler`
   - Addressing class imbalance using **SMOTEENN**

2. **Model Training & Hyperparameter Tuning**:

   - **Model**: XGBoost (Best performance)
   - **Hyperparameter tuning**: GridSearchCV

3. **Evaluation Metrics**:

   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - ROC-AUC Curve

4. **Model Explainability**:

   - Feature importance using `xgboost.plot_importance()`

## 📊 Results

XGBoost achieved an accuracy of **83.29% (without hyperparameter tuning)**,

After hyperparameter tuning, Accuracy: **95.7% (final)**

## 📌 How to Run Locally

1. Clone the repository
2. Install dependencies
3. Run the Jupyter Notebook or Python script to train the model.

## 🏆 Key Takeaways

✅ XGBoost performed best with **95.7% accuracy**\
✅ Proper handling of imbalanced data improves performance significantly

## 📌 Future Improvements

- Experiment with **Deep Learning models** (ANNs)
- Deploy as a **Flask/Django API** for integration with real applications
- Optimize performance using more advanced feature engineering


Feel free to ⭐ this repository if you found it useful! 😊

