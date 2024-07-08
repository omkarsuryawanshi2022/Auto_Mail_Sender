import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def Advertising(filename, features, target, test_size=0.2, random_state=42):

    # Load the dataset
    data = pd.read_csv(filename)
    
    # Display the first few rows of the dataset to understand its structure
    print(data.head())
    
    # Check for missing values (if any)
    print(data.isnull().sum())
    
    # Separate features (X) and target variable (y)
    X = data[features]
    y = data[target]
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Initialize the Linear Regression model
    model = LinearRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate R-squared
    r2 = r2_score(y_test, y_pred)
    print(f'R-squared value: {r2}')

def main():

    print("Marvellous Infosystem by Piyush Khairnar")
    print("Machine Learning Application")
    print("Advertising application using Linear Regression Algorithm")

    Advertising(filename='Advertising (1).csv', features=['TV', 'radio', 'newspaper'], target='sales')

if __name__ == "__main__":
    main()
