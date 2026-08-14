import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
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
X = data[['Hours_Studied']]
y = data['Exam_Score']

#splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

#training the linear regression model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

#making predictions on the test set
predictions = poly_model.predict(X_test_poly)

#evaluating the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}") # 12.352544212816404

r2 = r2_score(y_test, predictions)
print(f"R-squared: {r2}") # 0.2050784559067499

#visualizing the actual vs predicted scores
plt.scatter(y_test, predictions)
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs Predicted Scores')
plt.show()

#calculating correlation
print(f"Correlation between Hours Studied and Exam Score: {data['Hours_Studied'].corr(data['Exam_Score'])}") #--> 0.44510414026511613

print(f"Model Coefficients: {poly_model.coef_}")  # [0. , 0.2180706, 0.00176143]
print(f"Model Intercept: {poly_model.intercept_}") # 62.127391527300375

#----------thoughts on the model performance----------
# based on the model coefficients, it seems like the 2nd degree polynomial model isn't doing better than the linear model,
# as the quadratic term's coefficient is very small, indicating that the relationship between hours studied and exam score is still primarily linear.