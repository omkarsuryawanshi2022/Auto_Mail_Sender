# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# def MarvellousHeadBrainPredictor():

#     #Load data

#     data = pd.read_csv('MarvellousHeadBrain.csv')

#     print("Size of data set",data.shape)

#     X= data['Head Size(cm^3)'].values
#     Y= data['Brain Weight(grams)'].values

#     X= X.reshape((-1,1))

#     n = len(X)

#     reg = LinearRegression()

#     reg = reg.fit(X,Y)

#     print(r2)

# def main():

#     print("marvellous infosystem by piyush khairner")

#     print("Supervised mechine learning")

#     print("Linear Regrattion on Head Brain size data set")

#     MarvellousHeadBrainPredictor()

# if __name__ =="__main__":
#     main()


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score  # Import r2_score for calculating R-squared

def MarvellousHeadBrainPredictor():
    # Load data
    data = pd.read_csv('MarvellousHeadBrain.csv')

    print("Size of data set:", data.shape)

    X = data['Head Size(cm^3)'].values.reshape(-1, 1)  # Reshape X to 2D array (n_samples, n_features)
    Y = data['Brain Weight(grams)'].values

    # Create a linear regression model
    reg = LinearRegression()

    # Fit the model
    reg.fit(X, Y)

    # Make predictions
    Y_pred = reg.predict(X)

    # Calculate R-squared
    r2 = r2_score(Y, Y_pred)
    print("R-squared value:", r2)

    # Plotting the regression line (optional)
    import matplotlib.pyplot as plt
    plt.scatter(X, Y, color='blue', label='Actual data')
    plt.plot(X, Y_pred, color='red', linewidth=2, label='Linear regression')
    plt.xlabel('Head Size (cm^3)')
    plt.ylabel('Brain Weight (grams)')
    plt.legend()
    plt.show()

def main():
    print("Marvellous Infosystem by Piyush Khairner")
    print("Supervised Machine Learning")
    print("Linear Regression on Head Brain size dataset")
    MarvellousHeadBrainPredictor()

if __name__ == "__main__":
    main()
