import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def MarvellousPredictor():

    # Load data
    X = [1, 2, 3, 4, 5]
    Y = [3, 4, 2, 4, 5]

    print("Values of independent variable X:", X)
    print("Values of dependent variable Y:", Y)

    # Least square method
    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    print("Mean of independent variable X:", mean_x)
    print("Mean of dependent variable Y:", mean_y)
    n = len(X)

    numerator = 0
    denominator = 0

    # Equation of line is y = mx + c
    for i in range(n):
        numerator += (X[i] - mean_x) * (Y[i] - mean_y)
        denominator += (X[i] - mean_x) ** 2

    m = numerator / denominator

    # c = y - mx
    c = mean_y - (m * mean_x)

    print("Slope of the regression line is:", m)
    print("Y-intercept of the regression line is:", c)

    # Display plotting of above points
    x = np.linspace(1, 6, n)
    y = c + m * x

    plt.plot(x, y, color='#58b970', label='Regression Line')
    plt.scatter(X, Y, color='#ef5423', label='Scatter plot')

    plt.xlabel('X - Independent Variable')
    plt.ylabel('Y - Dependent Variable')

    plt.legend()
    plt.show()

    # Find out goodness of fit i.e., R Square
    ss_t = 0
    ss_r = 0

    for i in range(n):
        y_pred = c + m * X[i]
        ss_t += (Y[i] - mean_y) ** 2
        ss_r += (Y[i] - y_pred) ** 2

    r2 = 1 - (ss_r / ss_t)

    print("Goodness of fit using R^2 method is:", r2)


def main():
    print("Marvelos Infosystem by Piyush Khairner")
    print("Supervised Machine Learning")
    print("Linear Regression")

    MarvellousPredictor()


if __name__ == "__main__":
    main()
