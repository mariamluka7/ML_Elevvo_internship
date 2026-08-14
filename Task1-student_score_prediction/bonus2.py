import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# loading data from CSV file
data = pd.read_csv("Task1-student_score_prediction/StudentPerformanceFactors.csv")

#inspecting the data
# print(data.head())
# print(data.describe())
# print(data.info())
# print(data.isnull().sum())
# print(data.duplicated().sum())

#cleaning the data by dropping rows with missing values
data = data.dropna()

#visualizing the relationship between hours studied and exam score
# plt.scatter(data['Hours_Studied'], data['Exam_Score'])
# plt.xlabel('Hours Studied')
# plt.ylabel('Exam Score')
# plt.title('Relationship between Hours Studied and Exam Score')
# plt.show()

#defining features and target variable
X = data[[
    'Hours_Studied',
    'Attendance',
    'Sleep_Hours',
    'Previous_Scores',
    'Tutoring_Sessions',
    'Physical_Activity',
    ]]
y = data['Exam_Score']

#splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#making predictions on the test set
predictions = model.predict(X_test)

#evaluating the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}") 

RMSE = np.sqrt(mse)
print(f"Root Mean Squared Error: {RMSE}") 

r2 = r2_score(y_test, predictions)
print(f"R-squared: {r2}") 

#visualizing the actual vs predicted scores
plt.scatter(y_test, predictions)
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs Predicted Scores')
plt.show()

#calculating correlation
correlations = X.corrwith(y)
print("Correlations with Exam Score:")
print(correlations)

print(f"Model Coefficient: {model.coef_[0]}") 
print(f"Model Intercept: {model.intercept_}") 

#----------thoughts on the model performance----------
# adding multiple features ro the model improved it's performance,
# and that shows in the mse and r^2 values since they improved a lot.