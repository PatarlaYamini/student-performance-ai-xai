import pandas as pd

data = pd.read_csv("student-mat.csv", sep=";")

data["performance"] = pd.cut(
    data["G3"],
    bins=[-1, 9, 14, 20],
    labels=["Low", "Medium", "High"]
)

features = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup",
    "famsup", "paid", "activities", "nursery", "higher",
    "internet", "romantic", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences"
]

X = data[features]
y = data["performance"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns

numeric_features = X.select_dtypes(
    exclude=["object", "string"]
).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ))
    ]
)

model.fit(X_train, y_train)

print("Model trained successfully!")

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("This is our baseline model.")
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nFinal Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(classification_report(y_test, y_pred))


import pandas as pd

feature_names = model.named_steps["preprocessor"].get_feature_names_out()

importances = model.named_steps["classifier"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(importance_df.head(15))
# Test prediction for one student

new_student = pd.DataFrame([{
    "school": "GP",
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "A",
    "Medu": 3,
    "Fedu": 3,
    "Mjob": "teacher",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": 3,
    "failures": 0,
    "schoolsup": "yes",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 2,
    "Dalc": 1,
    "Walc": 1,
    "health": 3,
    "absences": 2
}])

prediction = model.predict(new_student)

print("\nPredicted Performance:", prediction[0])
import shap

explainer = shap.TreeExplainer(
    model.named_steps["classifier"]
)

print("SHAP explainer created successfully!")
transformed_student = model.named_steps["preprocessor"].transform(new_student)

shap_values = explainer.shap_values(transformed_student)

print("SHAP values calculated successfully!")
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

shap_values_array = shap_values[0]

if len(shap_values_array.shape) == 2:
    shap_values_array = shap_values_array[:, 0]

shap_importance = pd.DataFrame({
    "Feature": feature_names,
    "SHAP_Importance": abs(shap_values_array)
}).sort_values(
    by="SHAP_Importance",
    ascending=False
)

print("\nTop factors influencing this prediction:")
for feature in shap_importance["Feature"].head(10):
    clean_name = feature.replace("num__", "").replace("cat__", "")

    if "_" in clean_name:
        clean_name = clean_name.split("_")[0]

    print("-", clean_name)