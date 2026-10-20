# LAB EXERCISE 2 - DATA PREPROCESSING
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, Binarizer

df = pd.DataFrame({
    "Age": [20, 21, None, 23, 22],
    "Marks": [75, 82, 68, None, 91],
    "Department": ["IT", "CSE", "IT", "ECE", "CSE"]
})
print("Original data:\n", df)

# Handle missing values
for col in df.columns:
    if df[col].isna().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

print("\nAfter missing-value handling:\n", df)

# Encode categorical column
encoder = LabelEncoder()
df["Department_Encoded"] = encoder.fit_transform(df["Department"])
print("\nAfter label encoding:\n", df)

numeric = df[["Age", "Marks"]]
print("\nMin-Max scaling:\n", pd.DataFrame(
    MinMaxScaler().fit_transform(numeric), columns=numeric.columns
))
print("\nStandardization:\n", pd.DataFrame(
    StandardScaler().fit_transform(numeric), columns=numeric.columns
))
print("\nBinarization (threshold = 75):\n", pd.DataFrame(
    Binarizer(threshold=75).fit_transform(numeric), columns=numeric.columns
))
