Predictive Maintenance using Machine Learning
Project Overview

This project focuses on Predictive Maintenance, where the goal is to predict equipment failure before it actually happens. By analyzing historical data with machine learning models, we can identify patterns and anticipate failures, reducing downtime and maintenance costs.

Key Objectives

Understand machine failure data.

Perform data preprocessing, EDA, and visualization.

Apply ML classification models to predict failures.

Evaluate models with metrics such as Accuracy, Precision, Recall, F1-score, and ROC-AUC.

Save the final trained model for deployment.

Tech Stack

Python

Pandas, NumPy for data manipulation

Matplotlib, Seaborn for visualization

Scikit-learn for ML models

SMOTE for handling imbalanced data

Dataset

The dataset consists of equipment sensor readings and machine status (failure or healthy).

Target variable: Failure (0 = Normal, 1 = Failure).

Exploratory Data Analysis (EDA)

Visualized feature distributions.

Checked class imbalance → Failures were much less frequent.

Used SMOTE to balance the dataset.

Machine Learning Models Applied

Logistic Regression

Random Forest Classifier

Support Vector Machine (SVM)

Best Model

Random Forest Classifier achieved the best performance after hyperparameter tuning.

Results

Accuracy: ~100%

f1 score : ~100%

The model successfully predicts failures with good balance between precision & recall.

Saved Model

Final model is saved as final_model.pkl for future use.

How to Use

Clone the repo:

git clone https://github.com/your-username/predictive-maintenance.git
cd predictive-maintenance


Install dependencies:

pip install -r requirements.txt


Run Jupyter Notebook:

jupyter notebook predictive_maintenance.ipynb


Load and use trained model:

import joblib
model = joblib.load("final_model.pkl")
prediction = model.predict([[your_input_features]])

Future Improvements

Deploy the model with Flask/Django/Streamlit.

Integrate real-time IoT sensor data.

Use deep learning models for better performance.

👤 Author

Harshil Sutariya – Electrical Engineer & Aspiring Data Scientist 🚀
