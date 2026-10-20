# LAB EXERCISE 18 - SUPPORT VECTOR MACHINE (SVM)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 5, 7, 3],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 90, 58, 72, 85, 62],
    "Result": ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass",
               "Pass", "Pass", "Fail", "Pass", "Pass", "Fail"]
})
X, y = df[["Hours", "Attendance"]], df["Result"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Kernel: RBF")
print("Predictions:", predictions)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))
print("Support vectors per class:", model.named_steps["svc"].n_support_)
