# import numpy as np
# import pandas as pd
# from sklearn import metrics
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plot

# def MarvellousAdvertisementPredictor(data_path):
#     data = pd.read_csv(data_path, index_col=0)

#     print("Size of actual dataset ",len(data))

#     features_names = ['TV','radio','newspaper']

#     print("names of features",features_names)

#     X = data[features_names]

#     y = data.sales

#     X_train,Xtest,y_train,y_test = train_test_split(X,y,test_size=1/2)

#     print("size of Tranning dataset ", len(X_train))

#     print("size of Testing data set", len(X_test))

#     linreg = LinearRegression()

#     linreg.fit(X_train,y_train)

#     y_pred = linreg.predict(X_test)

#     print("Testing set")
#     print(X_test)

#     print("Result of Testing :")
#     print(y_pred)

#     print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

# def main():

#     MarvellousAdvertisementPredictor("Advertising (1).csv")

# if __name__ == "__main__":
#     main()

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def MarvellousAdvertisementPredictor(data_path):
    data = pd.read_csv(data_path, index_col=0)

    print("Size of actual dataset: ", len(data))

    features_names = ['TV', 'radio', 'newspaper']

    print("Names of features:", features_names)

    X = data[features_names]
    y = data['sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    print("Size of Training dataset:", len(X_train))
    print("Size of Testing dataset:", len(X_test))

    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    y_pred = linreg.predict(X_test)

    print("Testing set:")
    print(X_test)

    print("Predicted results:")
    print(y_pred)

    print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

def main():
    MarvellousAdvertisementPredictor("Advertising (1).csv")

if __name__ == "__main__":
    main()
