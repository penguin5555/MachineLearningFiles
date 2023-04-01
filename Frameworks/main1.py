import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("Frameworks\Train1.csv")

X_train = pd.get_dummies(df.drop(["PassengerId", "Survived", "Name"], axis=1))
y_train = df["Survived"].apply(lambda x: 1 if x==1 else 0)

df = pd.read_csv("Frameworks\Test1.csv")

X_test = pd.get_dummies(df.drop(["PassengerId", "Name"], axis=1))

