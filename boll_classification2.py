from sklearn import tree

# Rough=1
# smooth=0

# tennis 1
# cricket 2
def main():

    print("Ball classification case study")
    # Feature Encoding
    Features = [[35,1], [47,1], [90,0], [48,1],
                [90,0], [35,1], [92,0], [35,1], [35,1], [35,1]]

    Labels = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1]  # Corrected number of labels

    # Train the model
    obj = tree.DecisionTreeClassifier()

    # Train the model
    obj = obj.fit(Features, Labels)

    print(obj.predict([[96,0]]))
    print(obj.predict([[43,1]]))

if __name__ == "__main__":
    main()
