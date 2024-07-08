
from sklearn import tree

def Marvellous(weight,surface):


    # Features and labels
    Features = [[35,"Rough"], [47,"Rough"], [90,"smooth"], [48,"Rough"],
                [90,"smooth"], [35,"Rough"], [92,"smooth"], [35,"Rough"],
                [35,"Rough"], [35,"Rough"]]

    Labels = ["Tennis","Tennis","Cricket","Tennis","Cricket",
              "Tennis","Cricket","Tennis","Tennis","Tennis"]

    # Training a decision tree classifier
    clf = tree.DecisionTreeClassifier()
    clf=clf.fit(BallsFeatures, Labels)

    result=clf.predict([[weight,surface]])

    if result==1:
        print("your object lools like Teniis boll")
    elif result==2:
        print("your object lools like Cricket boll")


def main():
    print("Marvrllous infosystem by piyush khairnar")

    print("enter the weight of object")

    weight=input()

    print("what is the surface  type of your object Rough or smooth")

    surface=input()

    if surface.lower()=="rough":

        surface=1
    elif surface.lower()=="smooth":
        surface=0

    else:
        print("Error:Wrong input")
        exit()

    Marvellous(weight,surface)

if __name__ == "__main__":
    main()
