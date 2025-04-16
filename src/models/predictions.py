import pickle
import numpy as np


def diabetes_prediction(X_input: np.ndarray) -> tuple:
    model_filepath = "src/models/diabetes_model.pkl"
    with open(model_filepath, "rb") as file:
        pipeline = pickle.load(file)

    threshold_filepath = "src/models/model_threshold.pkl"
    with open(threshold_filepath, "rb") as file:
        threshold = pickle.load(file)

    """
    Predict outcomes for the given input data using the loaded model and threshold.

    Args:
        X_input (numpy array): Input data for prediction.

    Returns:
        tuple: A tuple containing the probability and binary prediction.
    """
    # Predict probabilities using the loaded model
    y_pred_proba = pipeline.predict_proba(X_input)

    # Apply the threshold to classify probabilities
    y_pred_adjusted = (y_pred_proba[:, 1] >= threshold).astype(float)

    # Return both the probability and the binary prediction
    return y_pred_proba[:, 1], y_pred_adjusted
