from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def main():
    print("Iris flower case study----------")

    data = load_iris()

    Feature = data.data
    Labels = data.target
    
    # Splitting the data into training and testing sets
    data_train, data_test, target_train, target_test = train_test_split(Feature, Labels, test_size=0.5)

    # Creating and training the decision tree classifier
    obj = tree.DecisionTreeClassifier()
    obj = obj.fit(data_train, target_train)

    # Making predictions on the test data
    output = obj.predict(data_test)

    print(output)

if __name__ == "__main__":
    main()
