# 💼 Employee Attrition Prediction App

An end-to-end Machine Learning project that predicts employee churn risk using demographic, compensation, and workplace satisfaction data. Built with Python, Scikit-Learn, and Streamlit.

---

## 📌 Project Overview
Employee attrition is a major cost for organizations. This project builds a binary classification model to predict whether an employee is likely to leave the company, helping HR teams take proactive retention measures.

* **Target Variable:** `Attrition` (1 = Yes, 0 = No)
* **Dataset:** IBM HR Analytics Employee Attrition & Performance
* **Key Challenge Handled:** Solved severe class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).
* **Model Choice:** Trained a **Random Forest Classifier** to maximize predictive accuracy and feature importance insights.

---

## 🛠️ Project Structure

```text
├── app.py              # Streamlit web interface for real-time predictions
├── train.py            # Model training, preprocessing, and model saving pipeline
├── requirements.txt    # Python dependencies required to run the project
└── README.md           # Project documentation
