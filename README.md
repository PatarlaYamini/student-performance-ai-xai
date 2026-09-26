# Student Performance Prediction System

An AI-powered machine learning web application that predicts student academic performance as Low, Medium, or High and explains the factors influencing each prediction using Explainable AI (SHAP).

## 1. Project Overview

This project uses student demographic, family, and study-related information to predict academic performance.

A Random Forest classifier predicts the student's performance category, while SHAP (SHapley Additive exPlanations) helps explain which input features contributed most to the prediction.

This project is intended for educational demonstration and should not be used to make final judgments about students.

## 2. Features

* Predicts student performance in three categories: Low, Medium, and High.
* Accepts 31 student input features.
* Provides predictions through a Flask web application.
* Displays the five largest absolute SHAP contributions for the predicted class.
* Includes a styled HTML and CSS interface.
* Uses preprocessing and one-hot encoding for categorical features.

## 3. Technologies Used

* Python
* Pandas and NumPy
* Scikit-learn
* SHAP
* Flask
* HTML and CSS
* Git and GitHub

## 4. Dataset

This project uses the Student Performance dataset from the UCI Machine Learning Repository, specifically `student-mat.csv`.

Dataset source:
https://archive.ics.uci.edu/dataset/320/student+performance

The dataset contains student demographic information, family background, study habits, school support, and academic grades.

### Target Categories

The final grade (`G3`) is converted into three performance categories:

| Category | Grade range |
| -------- | ----------- |
| Low      | 0–9         |
| Medium   | 10–14       |
| High     | 15–20       |

The previous period grades (`G1` and `G2`) are excluded from the model inputs to reduce target leakage.

The dataset is not included in this GitHub repository. Download the UCI Student Performance dataset from the source above and place `student-mat.csv` in the project root folder, alongside `app.py`.

## 5. Machine Learning Model

The project uses a Random Forest Classifier with a preprocessing pipeline.

* Categorical features are encoded using one-hot encoding.
* The dataset is split into training and testing sets using an 80:20 split.
* Stratification is used to preserve class proportions.
* The model uses class weighting to help address class imbalance.

### Model Evaluation

The baseline model was evaluated on a held-out test set.

**Held-out test accuracy: 53.16%**

| Category | Precision | Recall | F1-score |
| -------- | --------: | -----: | -------: |
| High     |      0.47 |   0.47 |     0.47 |
| Low      |      0.52 |   0.50 |     0.51 |
| Medium   |      0.56 |   0.58 |     0.57 |

These results are from a single train-test split and do not guarantee performance on new students or populations.

The Flask demonstration trains its model using the available dataset when the application starts. The held-out evaluation result above comes from the separate evaluation script and should not be interpreted as a guarantee of the web application's predictions.

## 6. Explainable AI (SHAP)

SHAP is used to estimate how input features contribute to individual model predictions.

The application displays the five features with the largest absolute SHAP contributions for the predicted class.

These values represent contribution magnitudes. They do not necessarily indicate whether a feature increases or decreases predicted performance, and they do not establish cause and effect.

## 7. Project Structure

```text
student_final/
│
├── app.py
├── student_prediction.py
├── student-mat.csv       # Download separately; not tracked by Git
├── README.md
├── .gitignore
│
└── templates/
    └── index.html
```

## 8. Installation and Setup

### Prerequisites

* Python installed on your computer
* Git
* Internet access to download the dataset and install Python packages

### Step 1: Clone the repository

```bash
git clone https://github.com/PatarlaYamini/student-performance-ai-xai.git
```

Move into the project folder:

```bash
cd student-performance-ai-xai
```

### Step 2: Download the dataset

Download the UCI Student Performance dataset from:

https://archive.ics.uci.edu/dataset/320/student+performance

Extract the downloaded files and copy `student-mat.csv` into the project folder, alongside `app.py`.

The application requires this file to train the model.

### Step 3: Install the required packages

On Windows, run:

```bash
py -m pip install pandas numpy scikit-learn shap flask
```

### Step 4: Start the web application

Run:

```bash
py app.py
```

The application trains the model and creates the SHAP explainer when it starts.

### Step 5: Open the application

Open this address in your browser:

http://127.0.0.1:5000

Keep the terminal running while using the application. To stop the local server, press `Ctrl + C` in the terminal.

## 9. Future Improvements

* Improve model performance through systematic evaluation and tuning.
* Add more detailed and user-friendly SHAP visualizations.
* Improve input validation and error handling.
* Add automated tests.
* Explore responsible deployment and model monitoring.

## 10. Disclaimer

This project is for educational and demonstration purposes only. Predictions are estimates based on historical data and should not be used as the sole basis for educational decisions.

Student performance depends on many factors that may not be represented in the dataset. The model may produce inaccurate or biased predictions and should not be used to label, rank, or make consequential decisions about students.

## 11. Dataset Attribution

Dataset: Student Performance
Source: UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/320/student+performance

Please refer to the dataset's official page for its citation and usage terms.
