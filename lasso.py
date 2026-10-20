# LAB EXERCISE 10 - LASSO REGRESSION
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = pd.DataFrame({
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 2, 5, 7, 3],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 90, 58, 72, 85, 62],
    "Marks": [35, 40, 45, 50, 60, 65, 75, 85, 42, 63, 82, 47]
})
X = df[["Hours_Studied", "Attendance"]]
y = df["Marks"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=10000))
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Lasso coefficients:", model.named_steps["lasso"].coef_)
print("Predictions:", predictions.round(2))
print("MSE:", mean_squared_error(y_test, predictions))
print("R-squared:", r2_score(y_test, predictions))
