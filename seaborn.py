# LAB EXERCISE 4 - SEABORN VISUALIZATION
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 90],
    "Marks": [35, 40, 45, 50, 60, 65, 75, 85],
    "Result": ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass"]
})

sns.set_theme(style="whitegrid")
sns.pairplot(df, hue="Result")
plt.show()

plt.figure(figsize=(7, 5))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

for column in ["Hours_Studied", "Attendance", "Marks"]:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.show()
