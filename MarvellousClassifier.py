from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

class MArvellousKNN():
    def fit(self, Training_Data, Training_Target):
        self.Training_Data = Training_Data
        self.Training_Target = Training_Target

    def predict(self, Test_Data):
        predictions = []
        for row in Test_Data:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        bestdistance = self.edu(row, self.Training_Data[0])
        bestindex = 0
        for i in range(1, len(self.Training_Data)):
            dist = self.edu(row, self.Training_Data[i])
            if dist < bestdistance:
                bestdistance = dist
                bestindex = i
        return self.Training_Target[bestindex]

    def edu(self, a, b):
        return distance.euclidean(a, b)

def main():
    border = "-" * 50

    iris = load_iris()

    data = iris.data
    target = iris.target

    print(border)
    print("Actual data set")
    print(border)
    for i in range(len(iris.target)):
        print("ID: %d, Label %s, Feature : %s" % (i, iris.data[i], iris.target[i]))
        print("Size of Actual data set %d" % (i + 1))

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.5)

    print(border)
    print("Training Data Set")
    print(border)
    for i in range(len(data_train)):
        print("ID: %d, Label %s, Feature : %s" % (i, data_train[i], target_train[i]))
        print("Size of Training data set %d" % (i + 1))

    print(border)
    print("Test Data Set")
    print(border)
    for i in range(len(data_test)):
        print("ID: %d, Label %s, Feature : %s" % (i, data_test[i], target_test[i]))
        print("Size of Test data set %d" % (i + 1))
        print(border)

    classifier = MArvellousKNN()
    classifier.fit(data_train, target_train)
    predictions = classifier.predict(data_test)
    accuracy = accuracy_score(target_test, predictions)

    print("Accuracy of Classification Algorithm with K Neighbor classifier is", accuracy * 100, "%")

if __name__ == "__main__":
    main()
