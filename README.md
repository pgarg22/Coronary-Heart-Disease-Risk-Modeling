
# Coronary Heart Disease Data Analysis

This repository contains data analysis and machine learning models applied to the **CHD (Coronary Heart Disease)** dataset.

## 📁 Files

### 1. `ExploratoryDataAnalysis.py`
Performs preprocessing and exploratory data analysis (EDA) on the CHD dataset. Key operations include:
- Data cleaning and transformation
- Converting categorical features into numerical ones
- Feature standardization
- Dimensionality reduction using **Principal Component Analysis (PCA)**
- Visualization of principal components and variance

### 2. `ClassificationAndRegressionModels.py`
Trains and compares several machine learning models for predicting outcomes from the CHD dataset. Includes:
- Models: Linear Regression, Ridge Regression, Logistic Regression, Decision Trees, Neural Networks
- Recursive Feature Elimination (RFE)
- Model performance evaluation with cross-validation
- Statistical significance testing (dependent t-tests)
- Neural network training and visualization using PyTorch

### 3. `CHD.csv`
The dataset used for analysis and modeling. Contains medical and demographic variables related to coronary heart disease.

## 🧠 Key Concepts
- Principal Component Analysis (PCA)
- Supervised learning (regression and classification)
- Feature selection
- Model comparison and evaluation
- Neural network training with PyTorch

## 📊 Tools & Libraries
- Python 3
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- PyTorch
- SciPy

## 📌 Authors
- Pranjal Garg
- Piotr Saffrani
- Avi Raj

## 🗃️ How to Run
Ensure that `CHD.csv` is in the same directory as the scripts. You can run each Python script using:

```bash
python ExploratoryDataAnalysis.py
python ClassificationAndRegressionModels.py
```

## 📜 License
This project is for educational purposes and part of a university coursework.

