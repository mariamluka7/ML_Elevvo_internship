import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("Task1-student_score_prediction/StudentPerformanceFactors.csv")

print(data.head())
print(data.describe())
print(data.info())
print(data.isnull().sum())
print(data.duplicated().sum())

data = data.dropna()

plt.scatter(data['Hours_Studied'], data['Exam_Score'])
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Relationship between Hours Studied and Exam Score')
plt.show()

X = data[['Hours_Studied']]
y = data['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}") #--> 12.351733799445533

r2 = r2_score(y_test, predictions)
print(f"R-squared: {r2}") #--> 0.20513060832466712

plt.scatter(y_test, predictions)
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs Predicted Scores')
plt.show()

print(f"Correlation between Hours Studied and Exam Score: {data['Hours_Studied'].corr(data['Exam_Score'])}") #--> 0.44510414026511613

print(f"Model Coefficient: {model.coef_[0]}") #--> 0.28834271
print(f"Model Intercept: {model.intercept_}") #--> 61.48999442610166