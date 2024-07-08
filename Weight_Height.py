import numpy as np
from sklearn.linear_model import LinearRegression

# Load data skipping the header
data = np.loadtxt('weight-height.csv', delimiter=',', skiprows=1, dtype=str)

# Extract features (x) and labels (y)
x_gender = data[:, 0]  # Gender column
x_height = data[:, 1].astype(float)  # Height column

# Convert categorical gender to numeric values (0 for Male, 1 for Female)
x_gender_numeric = np.where(x_gender == 'Male', 0, 1)

# Combine features into x
x = np.column_stack((x_gender_numeric, x_height))

# Extract labels (assuming the third column is the target variable, weight)
y = data[:, 2].astype(float)  # Weight column

# Fit linear regression model
reg = LinearRegression()
fitted_model = reg.fit(x, y)

# Make a prediction
# Note: When predicting, you need to provide the same format of data as during training
prediction_result = reg.predict([[0, 70]])  # Predicting for a Male with height 70

print("Prediction:", prediction_result)
