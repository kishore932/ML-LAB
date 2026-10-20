# LAB EXERCISE 1 - SIMPLE STATISTICS
import pandas as pd

# Replace this sample data with the dataset given by your instructor.
df = pd.DataFrame({
    "Marks": [45, 56, 67, 72, 81, 56, 90, 67, 74, 62]
})

print("Dataset:\n", df)
print("\nMean:\n", df.mean(numeric_only=True))
print("\nMedian:\n", df.median(numeric_only=True))
print("\nMode:\n", df.mode(numeric_only=True))
print("\nVariance:\n", df.var(numeric_only=True))
print("\nStandard deviation:\n", df.std(numeric_only=True))
print("\nMinimum:\n", df.min(numeric_only=True))
print("\nMaximum:\n", df.max(numeric_only=True))
print("\nSummary statistics:\n", df.describe())
