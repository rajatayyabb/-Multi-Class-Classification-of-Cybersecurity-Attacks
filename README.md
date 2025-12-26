# 🛡️ Multi-Class Classification of Cybersecurity Attacks

This project implements a machine learning pipeline to classify network traffic activity. By focusing on the **Top 15 most frequent attack types**, the model achieves high precision while filtering out rare noise.

## 🚀 Key Features
- **Data Preprocessing**: Comprehensive handling of missing values, scaling, and categorical encoding.
- **Top 15 Focus**: Simplifies the target variable to the most impactful attack categories for better model interpretability.
- **Multi-Model Approach**: Compares Random Forest, Logistic Regression, and XGBoost.
- **Interactive Dashboard**: Built with Streamlit for real-time traffic classification.

## 📁 Repository Structure
- `streamlit_app.py`: Main web application.
- `requirements.txt`: List of Python dependencies.
- `*.pkl`: Pre-trained models and preprocessing objects (Scaler, Encoders).

## 🛠️ Setup Instructions
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
