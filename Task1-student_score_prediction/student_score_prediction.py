import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# loading data from CSV file
data = pd.read_csv("Task1-student_score_prediction/StudentPerformanceFactors.csv")

#inspecting the data
print(data.head())
print(data.describe())
print(data.info())
print(data.isnull().sum())
print(data.duplicated().sum())

#cleaning the data by dropping rows with missing values
data = data.dropna()

#visualizing the relationship between hours studied and exam score
plt.scatter(data['Hours_Studied'], data['Exam_Score'])
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Relationship between Hours Studied and Exam Score')
plt.show()

#defining features and target variable
X = data[['Hours_Studied']]
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
print(f"Mean Squared Error: {mse}") #--> 12.351733799445533

r2 = r2_score(y_test, predictions)
print(f"R-squared: {r2}") #--> 0.20513060832466712

#visualizing the actual vs predicted scores
plt.scatter(y_test, predictions)
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs Predicted Scores')
plt.show()

#calculating correlation
print(f"Correlation between Hours Studied and Exam Score: {data['Hours_Studied'].corr(data['Exam_Score'])}") #--> 0.44510414026511613

print(f"Model Coefficient: {model.coef_[0]}") #--> 0.28834271
print(f"Model Intercept: {model.intercept_}") #--> 61.48999442610166

#----------thoughts on the model performance----------
# The model's R-squared value is relatively low (0.205), indicating that the- 
# model does not explain a large portion of the variance in exam scores based on hours studied alone. 
# This suggests that there are other factors influencing exam scores that are not captured by this simple linear regression model. 
# The correlation between hours studied and exam score is moderate (0.445), which indicates a positive relationship, 
# but it is not strong enough to make highly accurate predictions. Further feature engineering or the inclusion of additional-
# relevant variables may improve the model's performance.