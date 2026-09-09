# Explainable Ensemble Machine Learning for Credit Risk Classification in Banking Using SHAP-Based Interpretation

## Project Overview

This project was developed as part of the MSc Data Science and Artificial Intelligence programme at Sheffield Hallam University.

The aim of the project is to develop an explainable credit-risk classification pipeline that combines machine learning performance with transparent and understandable explanations for banking and finance professionals.

The project compares traditional and ensemble machine learning models, addresses class imbalance using SMOTE, evaluates model performance and asymmetric business costs, applies SHAP explainability, and evaluates the explanations through User Acceptance Testing (UAT).

## Dataset

The project uses the UCI Statlog German Credit dataset.

- 1,000 credit applicants
- 20 input features
- 700 good-credit applicants
- 300 bad-credit applicants
- 13 categorical features
- 7 numerical features

The dataset is split using a stratified 80/20 train-test split.

## Project Workflow

The overall workflow is:

Dataset Acquisition → Data Preprocessing → Train/Test Split → SMOTE → Model Training → Model Evaluation → Regularisation → Cost-Sensitive Evaluation → Model Selection → SHAP Explainability → UAT

SMOTE is applied only to the training data to prevent data leakage. The held-out test set remains unchanged for final evaluation.

## Machine Learning Models

The following models were investigated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Regularised XGBoost
- Regularised LightGBM

Logistic Regression was used as a simpler baseline for comparison with the ensemble models.

## Model Evaluation

Model performance was evaluated using:

- F1-score
- Matthews Correlation Coefficient (MCC)
- AUC-ROC
- Train-test performance comparison
- Cost-sensitive evaluation

Regularised XGBoost achieved the strongest overall F1-score (0.6179) and MCC (0.4486) while reducing overfitting compared with the original XGBoost model.

## Cost-Sensitive Evaluation

Credit-classification errors can have different financial consequences. Therefore, the project also uses asymmetric cost-sensitive evaluation.

Total Expected Cost = (1 × Good → Bad) + (5 × Bad → Good)

Where:

- Good → Bad = incorrectly rejecting a good applicant (cost = 1)
- Bad → Good = incorrectly accepting a bad applicant (cost = 5)

Logistic Regression achieved the lowest expected cost of 132, while Regularised XGBoost achieved a slightly higher cost of 135.

This demonstrates a trade-off between overall predictive performance and asymmetric business cost.

## SHAP Explainability

Regularised XGBoost was selected for SHAP explainability because of its strong F1-score and MCC performance, reduced overfitting, and compatibility with TreeSHAP.

SHAP was used to provide both global and individual explanations.

The project includes:

- SHAP Waterfall Plots
- SHAP Beeswarm Plots
- SHAP Force Plots
- Global feature importance
- Individual applicant explanations

Checking account status was identified as the most influential feature in the fitted model.

## User Acceptance Testing

User Acceptance Testing was conducted with eight banking and finance professionals.

Participants evaluated SHAP explanations based on:

- Clarity
- Trustworthiness
- Usefulness

Because the sample size was small (n=8) and Likert-scale data are ordinal, results were primarily evaluated using response distributions.

Responses were predominantly concentrated in the Agree and Strongly Agree categories, indicating generally positive perceptions of the SHAP explanations.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- imbalanced-learn
- SMOTE
- SHAP
- Matplotlib
- Git
- GitHub

## Key Findings

Regularised XGBoost achieved the strongest overall F1-score and MCC while reducing overfitting.

Logistic Regression achieved the lowest asymmetric expected cost, showing that model selection can depend on both predictive performance and business consequences.

SHAP provided transparent global and individual explanations of credit-risk predictions.

UAT results showed that banking and finance professionals generally perceived the SHAP explanations as clear, trustworthy and useful.

## Conclusion

The project demonstrates that ensemble machine learning can be combined with SHAP explainability to provide strong credit-risk classification performance alongside transparent and practically useful explanations.

The findings also highlight the importance of considering class imbalance, overfitting, asymmetric business costs and stakeholder evaluation when developing explainable machine learning systems for credit-risk assessment.

## Academic Project

This repository contains work completed for an MSc Data Science and Artificial Intelligence Applied Project at Sheffield Hallam University.

The project is intended for academic and research purposes.