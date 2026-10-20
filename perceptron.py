# LAB EXERCISE 16 - PERCEPTRON MODEL
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Train a perceptron to learn the AND logic gate.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

model = Perceptron(eta0=0.1, max_iter=1000, tol=1e-4, random_state=42)
model.fit(X, y)
predictions = model.predict(X)

print("Input  Expected  Predicted")
for inputs, expected, predicted in zip(X, y, predictions):
    print(f"{inputs}     {expected}         {predicted}")
print("Weights:", model.coef_)
print("Bias:", model.intercept_)
print("Accuracy:", accuracy_score(y, predictions))
