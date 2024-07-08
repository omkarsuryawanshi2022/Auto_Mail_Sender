import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

def main():
    print("Ball classification case study")

    # Features and labels
    Features = [[35,"Rough"], [47,"Rough"], [90,"smooth"], [48,"Rough"],
                [90,"smooth"], [35,"Rough"], [92,"smooth"], [35,"Rough"],
                [35,"Rough"], [35,"Rough"]]

    Labels = ["Tennis","Tennis","Cricket","Tennis","Cricket",
              "Tennis","Cricket","Tennis","Tennis","Tennis"]

    # Encoding categorical feature
    encoder = LabelEncoder()
    Features_encoded = [[feature[0], encoder.fit_transform([feature[1]])[0]] for feature in Features]

    # Training a decision tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(Features_encoded, Labels)

    # Predicting the class of a new sample
    prediction = clf.predict([[96, encoder.transform(["Rough"])[0]]])
    print("Predicted class:", prediction)

if __name__ == "__main__":
    main()
