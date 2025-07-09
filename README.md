# 🍷 Wine Quality Classification using XGBoost

This project uses machine learning to classify the quality of red wine into binary categories (good vs not good) based on its physicochemical properties. The model is trained using the powerful XGBoost algorithm with hyperparameter tuning and SMOTE for class imbalance.

---

## 📁 Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **File**: `winequality-red.csv`
- **Target Variable**: `quality` (converted to binary: 1 = good quality (>=7), 0 = otherwise)

---

## 📊 Features Used

- Fixed Acidity  
- Volatile Acidity  
- Citric Acid  
- Residual Sugar  
- Chlorides  
- Free Sulfur Dioxide  
- Total Sulfur Dioxide  
- Density  
- pH  
- Sulphates  
- Alcohol

---

## 🔍 Project Workflow

1. **Data Loading & Cleaning**
   - Removed duplicate entries.
   - Removed outliers using IQR method.

2. **Target Conversion**
   - Converted multi-class `quality` into binary:  
     `1` (good wine if quality ≥ 7) and `0` (otherwise).

3. **Train-Test Split**
   - Used `train_test_split` with `stratify=y` for class balance.

4. **Feature Scaling**
   - Applied `StandardScaler` for normalization.

5. **Handling Class Imbalance**
   - Used **SMOTE (Synthetic Minority Over-sampling Technique)** on training data.

6. **Model Building**
   - Applied `XGBClassifier` with parameter tuning using `RandomizedSearchCV`.

7. **Model Evaluation**
   - Metrics: Accuracy, Confusion Matrix, Classification Report.
   - Feature importance plotted.

---

## 🛠️ Technologies Used

- Python 🐍
- Pandas, NumPy
- Seaborn, Matplotlib
- scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)

---

## 📈 Results

- Final model achieved high accuracy and good balance across precision and recall.
- SMOTE improved the model's ability to detect minority class (good wines).
- Alcohol, Sulphates, and Volatile Acidity were among the most important features.

---

## 📂 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/wine-quality-xgboost.git
   cd wine-quality-xgboost
