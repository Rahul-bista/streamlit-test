# imports 

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, r2_score
import joblib

df = pd.read_csv("student_depression_dataset.csv")
df.dropna(inplace=True)
df['Financial Stress'] = pd.to_numeric(df['Financial Stress'], errors='coerce')

df['Gender'] = pd.factorize(df['Gender'])[0]
df['Sleep Duration'] = pd.factorize(df['Sleep Duration'])[0]
df['Dietary Habits'] = pd.factorize(df['Dietary Habits'])[0]
df['Degree'] = pd.factorize(df['Degree'])[0]
df['Have you ever had suicidal thoughts ?'] = pd.factorize(df['Have you ever had suicidal thoughts ?'])[0]
df['Family History of Mental Illness'] = pd.factorize(df['Family History of Mental Illness'])[0]

# Store Predictor and target on X & y

X = df.drop(columns=["id", "City", "Work Pressure", "Profession", "Job Satisfaction", "Depression"])
y = df["Depression"]

# Create model instance

model = RandomForestClassifier(random_state=16)

# Split data for training and testing for predictor and target

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=16)

# Train the Model

model = model.fit(train_x, train_y)

# Testing the model for wtv reason

predict = model.predict(test_x)

joblib.dump(model, "depression_model.pkl")
