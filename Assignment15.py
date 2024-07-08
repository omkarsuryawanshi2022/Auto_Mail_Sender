from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def WinePredictor():
    # Load dataset
    wine = datasets.load_wine()

    # Print the name of the features
    print(wine.feature_names)

    # Print the label species (class_0, class_1, class_2)
    print(wine.target_names)

    # Print the wine data (top 5 records)
    print(wine.data[:5])

    # Print the wine labels (0: class_0, 1: class_1, 2: class_2)
    print(wine.target)

    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)  # 70% training and 30% test

    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)  # Initialize the KNN classifier with k=5

    # Fit the classifier to the training data
    knn.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = knn.predict(x_test)

    # Model accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

def main():
    print("---- Marvellous Infosystem by Piyush Khairnar ----") 

    print("Machine Learning Application")

    print("Wine predictor application using k Nearest Neighbor Algorithm")
    WinePredictor()

if __name__ == "__main__":
    main()
