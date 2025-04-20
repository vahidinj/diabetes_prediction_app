import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    classification_report,
    log_loss,
    auc,
    precision_score,
    f1_score,
    recall_score,
)


# Load Data
df = pd.read_csv("Data/diabetes.csv")
df = df.drop(["BloodPressure", "SkinThickness"], axis=1)
# Droping Feature Outliers that worsen model performance
print("Initial rows:", len(df))
df = df[df["Insulin"] <= 300]
print("Rows after Insulin filter:", len(df))
df = df[df["Glucose"] > 0]
print("Rows after Glucose filter:", len(df))
df = df[(df["BMI"] < 55) & (df["BMI"] > 0)]
print("Rows after BMI filter:", len(df))
df = df[df["DiabetesPedigreeFunction"] <= 1.2]
print("Rows after DiabetesPedigreeFunction filter:", len(df))
df = df[df["Age"] < 70]
print("Rows after Age filter:", len(df))
df = df[df["Pregnancies"] < 15]
print("Rows after Pregnancies filter:", len(df))

# Model Building
X = np.asarray(df.drop("Outcome", axis=1))
y = np.asarray(df["Outcome"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", LogisticRegression())])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)


# MODEL EVALUATION

# Classification Report
# Classification Report as a DataFrame
class_report_dict = classification_report(
    y_pred=y_pred, y_true=y_test, output_dict=True
)
class_report_df = pd.DataFrame(class_report_dict).transpose()  # Convert to DataFrame
print(f"Classification Report DataFrame:\n{class_report_df}")

# Accuracy Score
score = accuracy_score(y_pred=y_pred, y_true=y_test)
print(f"Accuracy Score: {score}")

# Loss
loss = log_loss(y_test, y_pred_proba)
print(f"Log Loss: {loss}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


# Model Visualization
# ROC Curve and AUC


def roc_curve_graph():
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()

    ax.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    return fig


# roc_curve_graph()


thresholds = [range(0, 1)]


def evaluate_threshold(threshold):
    y_pred_adjusted = (y_pred_proba[:, 1] >= threshold).astype(int)
    # Convert the 0/1 predictions back into a 2D probability distribution
    y_pred_adjusted_proba = np.column_stack((1 - y_pred_adjusted, y_pred_adjusted))

    loss_adjusted = log_loss(y_test, y_pred_adjusted_proba)
    cm_adjusted = confusion_matrix(y_test, y_pred_adjusted)
    precision_adjusted = precision_score(y_test, y_pred_adjusted)
    recall_adjusted = recall_score(y_test, y_pred_adjusted)
    f1_adjusted = f1_score(y_test, y_pred_adjusted)
    accuracy_adjusted = (y_pred_adjusted == y_test).mean()

    print(f"Threshold: {threshold}")
    print(f"log_loss: {loss_adjusted}")
    print("Confusion Matrix (Adjusted):\n", cm_adjusted)
    print(f"Precision (Adjusted): {precision_adjusted}")
    print(f"Recall (Adjusted): {recall_adjusted}")
    print(f"F1-score (Adjusted): {f1_adjusted}")
    print(f"Accuracy (Adjusted): {accuracy_adjusted}")


# Evaluate model performance at different thresholds
for threshold in thresholds:
    evaluate_threshold(threshold)


# Best model considering that this is medically related is 0.45, despite 0.6 having the best accuracy and precision.

SELECTED_THRESHOLD = 0.45

y_pred_adjusted = (y_pred_proba[:, 1] >= SELECTED_THRESHOLD).astype(int)
# Convert the 0/1 predictions back into a 2D probability distribution
y_pred_adjusted_proba = np.column_stack((1 - y_pred_adjusted, y_pred_adjusted))


def compute_metrics(threshold):
    # Adjust predictions based on the threshold
    y_pred_adjusted = (y_pred_proba[:, 1] >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_adjusted)

    # Classification report as a DataFrame
    class_report = classification_report(y_test, y_pred_adjusted, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()

    # Accuracy and loss
    score = accuracy_score(y_test, y_pred_adjusted)
    loss = log_loss(y_test, y_pred_proba)

    return cm, class_report_df, score, loss


# Exporting the model
model_filename = "diabetes_model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(pipeline, file)

print(f"\nModel saved as {model_filename}")

threshold_filename = "model_threshold.pkl"
with open(threshold_filename, "wb") as file:
    pickle.dump(SELECTED_THRESHOLD, file)

print(f"Selected threshold saved as {threshold_filename}")
