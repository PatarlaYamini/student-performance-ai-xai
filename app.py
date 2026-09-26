from flask import Flask, render_template, request
import pandas as pd
import shap

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset
data = pd.read_csv("student-mat.csv", sep=";")

# Create performance categories
data["performance"] = pd.cut(
    data["G3"],
    bins=[-1, 9, 14, 20],
    labels=["Low", "Medium", "High"]
)

# Features used by the model
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

# Identify categorical and numerical features
categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns

numeric_features = X.select_dtypes(
    exclude=["object", "string"]
).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numeric_features
        )
    ]
)

# Model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

# Train model
model.fit(X, y)

print("31-feature model trained successfully!")
explainer = shap.TreeExplainer(
    model.named_steps["classifier"]
)

print("SHAP explainer created successfully!")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    shap_values = None
    shap_factors = None
    if request.method == "POST":

        student = {
            "school": request.form["school"],
            "sex": request.form["sex"],
            "age": int(request.form["age"]),
            "address": request.form["address"],
            "famsize": request.form["famsize"],
            "Pstatus": request.form["Pstatus"],
            "Medu": int(request.form["Medu"]),
            "Fedu": int(request.form["Fedu"]),
            "Mjob": request.form["Mjob"],
            "Fjob": request.form["Fjob"],
            "reason": request.form["reason"],
            "guardian": request.form["guardian"],
            "traveltime": int(request.form["traveltime"]),
            "studytime": int(request.form["studytime"]),
            "failures": int(request.form["failures"]),
            "schoolsup": request.form["schoolsup"],
            "famsup": request.form["famsup"],
            "paid": request.form["paid"],
            "activities": request.form["activities"],
            "nursery": request.form["nursery"],
            "higher": request.form["higher"],
            "internet": request.form["internet"],
            "romantic": request.form["romantic"],
            "famrel": int(request.form["famrel"]),
            "freetime": int(request.form["freetime"]),
            "goout": int(request.form["goout"]),
            "Dalc": int(request.form["Dalc"]),
            "Walc": int(request.form["Walc"]),
            "health": int(request.form["health"]),
            "absences": int(request.form["absences"])
        }

        new_student = pd.DataFrame([student])

        prediction = model.predict(new_student)[0]

        transformed_student = model.named_steps["preprocessor"].transform(new_student)

        shap_values = explainer.shap_values(transformed_student)

        # Find the class predicted by our model
        class_index = list(model.named_steps["classifier"].classes_).index(prediction)

        # Handle SHAP output formats
        if isinstance(shap_values, list):
            values = shap_values[class_index][0]
        else:
            values = shap_values

            if len(values.shape) == 3:
                values = values[0, :, class_index]
            else:
                values = values[0]

        # Get the names of the transformed features
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()

        # Combine one-hot encoded features into original features
        importance = {}

        for name, value in zip(feature_names, values):
            clean_name = name.replace("num__", "").replace("cat__", "")

            original_name = clean_name.split("_")[0]

            importance[original_name] = (
                importance.get(original_name, 0) + abs(float(value))
            )

        friendly_names = {
    "schoolsup": "Extra school support",
    "Fjob": "Father's job",
    "absences": "Number of absences",
    "goout": "Going out with friends",
    "Medu": "Mother's education",
    "Fedu": "Father's education",
    "studytime": "Study time",
    "failures": "Previous failures",
    "health": "Health status",
"freetime": "Free time",
"reason": "Reason for choosing school"
}
        shap_factors = sorted(
            importance.items(),
            key=lambda item: item[1],
            reverse=True
        )[:5]
        shap_factors = [
    (friendly_names.get(name, name), value)
    for name, value in shap_factors
]
    return render_template(
    "index.html",
    prediction=prediction,
    shap_values=shap_values,
    shap_factors=shap_factors
)
if __name__ == "__main__":
    app.run(debug=True)