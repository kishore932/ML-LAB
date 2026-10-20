# LAB EXERCISE 14 - NAIVE BAYES CLASSIFICATION
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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
model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Predictions:", predictions)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))
