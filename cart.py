# LAB EXERCISE 12 - DECISION TREE CART (GINI)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

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
model = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Predictions:", predictions)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))

plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=X.columns, class_names=model.classes_,
          filled=True, rounded=True)
plt.tight_layout()
plt.show()
