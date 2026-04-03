# Student Placement Prediction System

This project is a Streamlit-based web application that predicts student placement probabilities based on various factors like CGPA, internship experience, and technical skills.

## Features

- **Placement Prediction**: Predicts whether a student will be placed or not.
- **Probability Analysis**: Shows the likelihood of placement.
- **What-if Analysis**: Allows users to see how changing certain parameters (e.g., higher CGPA) affects the placement probability.
- **Open-source Explainability**: Uses SHAP to explain the model's predictions.
- **Personalized Recommendations**: Provides tips on how to improve placement chances.

## Project Structure

- `app.py`: The main Streamlit application.
- `student_placement_notebook_v2.ipynb`: Jupyter notebook used for data exploration and model training.
- `models/`: Contains the trained machine learning model and SHAP background data.
- `requirements.txt`: Python dependencies.
- `Sample.csv`: Dataset used for training/testing.

## How to Run

1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
