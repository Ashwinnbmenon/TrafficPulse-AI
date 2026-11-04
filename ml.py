import joblib
import pandas as pd
import numpy as np
from functools import lru_cache

MODEL_PATH = "models/best_traffic_model.pkl"


@lru_cache(maxsize=1)
def load_artifact():
    return joblib.load(MODEL_PATH)


def _prepare(input_data, artifact):
    data = pd.DataFrame([input_data]).reindex(columns=artifact["feature_columns"])

    for col in artifact["numeric_columns"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(artifact["numeric_fill_values"][col])

    for col in artifact["categorical_columns"]:
        encoder = artifact["label_encoders"][col]
        data[col] = data[col].fillna(
            artifact["categorical_fill_values"][col]
        ).astype(str)

        unknown = set(data[col]) - set(encoder.classes_)
        if unknown:
            raise ValueError(f"Unknown category in {col}: {unknown}")

        data[col] = encoder.transform(data[col])

    data = data[artifact["feature_columns"]]

    if artifact["model_name"] == "Logistic Regression":
        data = artifact["scaler"].transform(data)

    return data


def _class_labels(artifact):
    classes = artifact["model"].classes_
    if artifact["model_name"] == "XGBoost":
        return artifact["target_encoder"].inverse_transform(
            np.asarray(classes).astype(int)
        )
    return classes


def predict_congestion(input_data):
    artifact = load_artifact()
    data = _prepare(input_data, artifact)
    prediction = artifact["model"].predict(data)

    if artifact["model_name"] == "XGBoost":
        prediction = artifact["target_encoder"].inverse_transform(
            np.asarray(prediction).astype(int)
        )

    return str(prediction[0])


def predict_congestion_proba(input_data):
    """Returns (label, {class_label: probability})."""
    artifact = load_artifact()
    data = _prepare(input_data, artifact)
    model = artifact["model"]

    label = predict_congestion(input_data)

    if not hasattr(model, "predict_proba"):
        return label, {}

    probs = model.predict_proba(data)[0]
    return label, {str(c): float(p) for c, p in zip(_class_labels(artifact), probs)}