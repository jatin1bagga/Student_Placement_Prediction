# 🎓 Student Placement Prediction System

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest%20%7C%20XGBoost%20%7C%20LogReg-green.svg)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange.svg)

An end-to-end Machine Learning project and interactive web application built with **Streamlit**. This system predicts the likelihood of a student being placed based on their academic performance, demographics, and extracurricular activities.

## 🌟 Key Features

* **Real-time Placement Prediction**: Instantly predicts whether a student will be placed or not.
* **Placement Probability**: Displays a percentage indicating the confidence of the prediction.
* **Interactive What-If Analysis**: Tweak individual factors (like bumping up CGPA or adding an internship) to see exactly how much it changes your placement probability in real-time.
* **Open-Source Explainability (SHAP)**: Uses SHAP (SHapley Additive exPlanations) to provide waterfall and bar charts that explain exactly *why* the model made its prediction and which features had the biggest impact.
* **Personalized Recommendations**: A rule-based recommendation engine that gives actionable advice on how to improve placement chances based on the specific inputs.

## 🛠️ Technology Stack

* **Frontend/UI**: Streamlit
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn, XGBoost
* **Model Interpretation**: SHAP (SHapley Additive exPlanations)
* **Environment**: Python 3, `uv` / `pip`

## 📂 Project Structure

```text
Student_Placement_Prediction/
├── app.py                              # Main Streamlit web application
├── student_placement_notebook_v2.ipynb # Complete EDA, pre-processing, and model training notebook
├── Sample.csv                          # Dataset used for training and evaluation
├── requirements.txt                    # List of Python dependencies
├── .gitignore                          # Git ignore rules
├── models/                             # Serialized ML artifacts
│   ├── best_model.pkl                  # Trained Scikit-Learn pipeline (Preprocessor + Classifier)
│   └── shap_background.pkl             # Background dataset sample used for SHAP explainers
└── README.md                           # Project documentation
```

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/jatin1bagga/Student_Placement_Prediction.git
cd Student_Placement_Prediction
```

### 2. Set up a virtual environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
The application will be available in your browser at `http://localhost:8501`.

## 🧠 Model Details

The machine learning pipeline was trained and evaluated using Jupyter Notebook. The process included:
1. **Data Cleaning & Exploration**: Handling missing values, EDA, and understanding feature distributions.
2. **Feature Engineering**: Creating composite `profile_score` and `experience_index` features.
3. **Preprocessing**: Building a `ColumnTransformer` to handle One-Hot Encoding for categorical variables and Standard Scaling for numeric variables.
4. **Model Comparison**: Training Logistic Regression, Random Forest, and XGBoost models. 
5. **Selection**: The best performing model (based on F1-Score & Accuracy) was serialized and integrated into the Streamlit app.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
