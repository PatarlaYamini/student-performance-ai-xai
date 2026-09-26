# Student Performance Prediction System

An AI-powered machine learning web application that predicts student performance as Low, Medium, or High and explains the factors influencing each prediction.

## Project Overview

This project uses student information and study-related factors to predict academic performance. It also uses Explainable AI (SHAP) to identify factors that contributed to a prediction.

The system is intended for educational demonstration and should not be used to make final judgments about students.

## Features

* Predicts student performance in three categories: Low, Medium, and High.
* Accepts 31 student input features.
* Displays predictions through a Flask web application.
* Uses SHAP to show the five most influential factors for a prediction.
* Includes a simple, styled user interface.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* SHAP
* Flask
* HTML and CSS
* Git and GitHub

## Dataset

The project uses the UCI Student Performance dataset, specifically `student-mat.csv`.

The dataset contains information about students' demographics, family background, study habits, school support, and academic performance.

The final grade (G3) is converted into three categories:

| Category | Grade range |
| -------- | ----------- |
| Low      | 0–9         |
| Medium   | 10–14       |
| High     | 15–20       |

The previous period grades G1 and G2 are excluded from the model inputs to reduce target leakage.

## Machine Learning Model

The project uses a Random Forest Classifier with preprocessing and one-hot encoding for categorical features.

The dataset is split into training and testing sets using an 80:20 split, with stratification to preserve class proportions.

### Model Evaluation

Held-out test accuracy: **53.16%**

| Category | Precision | Recall | F1-score |
| -------- | --------: | -----: | -------: |
| High     |      0.47 |   0.47 |     0.47 |
| Low      |      0.52 |   0.50 |     0.51 |
| Medium   |      0.56 |   0.58 |     0.57 |

These results are based on a single train-test split and do not guarantee performance on new student populations.

## Explainable AI

SHAP (SHapley Additive exPlanations) is used to estimate the contribution of input features to individual predictions.

The application displays the five factors with the largest absolute SHAP contributions for the predicted class.

These values indicate contribution magnitude, not necessarily whether a factor increases or decreases predicted performance.

## How to Run the Project

1. Install Python.
2. Open a terminal in the project folder.
3. Install the required packages:

```bash
py -m pip install pandas numpy scikit-learn shap flask
```

4. Make sure `student-mat.csv` is in the project folder.
5. Start the web application:

```bash
py app.py
```

6. Open this address in your browser:

```text
http://127.0.0.1:5000
```

## Project Structure

```text
student_final/
│
├── app.py
├── student_prediction.py
├── student-mat.csv
├── README.md
│
└── templates/
    └── index.html
```

## Future Improvements

* Improve model performance through systematic model evaluation and tuning.
* Add more detailed and user-friendly SHAP visualizations.
* Improve input validation and error handling.
* Add automated tests.
* Explore responsible deployment and model monitoring.

## Disclaimer

This project is for educational and demonstration purposes only. Predictions are estimates based on historical data and should not be used as the sole basis for educational decisions.

## Dataset Source

UCI Machine Learning Repository — Student Performance dataset.

https://archive.ics.uci.edu/dataset/320/student+performance
