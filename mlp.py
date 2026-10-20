# LAB EXERCISE 17 - MULTI-LAYER PERCEPTRON (MLP)
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# An MLP learns the XOR logic gate, which is not linearly separable.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(4,), activation="logistic",
                  solver="lbfgs", max_iter=3000, random_state=42)
)
model.fit(X, y)
predictions = model.predict(X)

print("Input  Expected  Predicted")
for inputs, expected, predicted in zip(X, y, predictions):
    print(f"{inputs}     {expected}         {predicted}")
print("Accuracy:", accuracy_score(y, predictions))
