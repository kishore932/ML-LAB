# LAB EXERCISE 7 - FIND-S CONCEPT LEARNING
# Attributes: Sky, AirTemp, Humidity, Wind, Water, Forecast
training_data = [
    (["Sunny", "Warm", "Normal", "Strong", "Warm", "Same"], "Yes"),
    (["Sunny", "Warm", "High", "Strong", "Warm", "Same"], "Yes"),
    (["Rainy", "Cold", "High", "Strong", "Warm", "Change"], "No"),
    (["Sunny", "Warm", "High", "Strong", "Cool", "Change"], "Yes"),
]

hypothesis = None
for example, target in training_data:
    print("Example:", example, "Target:", target)
    if target == "Yes":
        if hypothesis is None:
            hypothesis = example.copy()
        else:
            for i in range(len(hypothesis)):
                if hypothesis[i] != example[i]:
                    hypothesis[i] = "?"

print("\nFinal most-specific hypothesis:", hypothesis)
