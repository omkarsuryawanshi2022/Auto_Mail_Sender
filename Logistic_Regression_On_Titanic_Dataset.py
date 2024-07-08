# import math
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from seaborn import countplot
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure, show
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression

# def MarvellousTitanicLogistic():
#     #step first data load
#     titanic_data = pd.read_csv('marvellousTitanicDataSet.csv')

#     print("First 5 entries from loaded dataset")
#     print(titanic_data.head())

#     print( "number of passangers are"+str(len(titanic_data)))

#     # step 2 Anylyse data

#     print("Visualisation: Supervised and non supervised passangers")
#     figure()
#     target = "survived"

#     countplot(data=titanic_data,x=target,hue="sex").set_title("Marvellous infosystem:survived and non survived passangers based on gender")
#     show()

#     print("Visualisation: Survived and non survied passangers basedon the passenger class")
#     figure()
#     target="Survived"

#     countplot(data=titanic_data,x=target,hue="pclass").set_title("Marvellos infosystem:survived and non survied passengers based on the pasenger class")
#     show()

#     print("visualisation: survived and non survived passengers based on age")
#     figure()
#     titanic_data["Age"].plot.hist().set_title("Marvellous infosystem: survivved and non survived passenger based on age")
#     show()

#     print("visualisation: survived and non survived passengers based on Fare")
#     figure()

#     titanic_data["Fare"].plot.hist().set_title("Marvellous Infosystems: survived and non survived passengers based on fare")
#     show()

#     # Step 3 Data Cleanning
#     titanic_data.drop("zero",axis = 1, inplace = True)

#     print("First 5 entries from loaded dataset after removing zero coloumns")
#     print(titanic_data.head(5))

#     print("values of sex columns")
#     print(pd.get_dummies(titanic_data["Sex"]))

#     print("values of sex columns after removing one  field")
#     Sex = pd.get_dummies(titanic_data["sex"], drop_first = True)
#     print(Sex.head(5))

#     print("values of sex columns after removing one  field")
#     pclass= pd.get_dummies(titanic_data["pclass"], drop_first = True)
#     print(pclass.head(5))

#     print("values of data set after concatenating new  columns")
#     titanic_data = pd.concat([titanic_data,Sex,pclass],axis = 1)
#     print(titanic_data.head(5))

#     print("values of data set after removing irrelevent  columns")
#     titanic_data.drop(["sex","sibsp","parch","Embarked"], axis = 1, inplace =True)
#     print(titanic_data.head(5))

#     x = titanic_data.drop("supervised",axis = 1)
#     y = titanic_data["survived"]

#     # step 4 : Data Tranning
#     xtrain,xtest ,ytrain, ytest = train_test_split(x,y,test_size=0.5)

#     logmodel = LogisticRegression()

#     logmodel.fit(xtrain,ytrain)

#     # step 4 Data Testing

#     prediction = logmodel.predict(xtest)

#     # step 5 Calculate Accuracy

#     print(" classification  report of logistic Regration is :")
#     print(Classification_report(ytest,prediction))

#     print("Confussion matrix of Logistic Regrassion is :")
#     print(confusion_matrix(ytest,prediction))

#     print("Accuracy of Logistic Regrassion is :")
#     print(accuracy_score(ytest,prediction))

# def main():

#     print("-----Marvellous infosystem by piyush khairner ------")

#     print("Supervised meachine learning ")

#     print("Logistic Regrassion on Titanic data set")

#     MarvellousTitanicLogistic()

# if __name__ == "__main__":
#     main()



# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression

# def MarvellousTitanicLogistic():
#     # Step 1: Load data
#     titanic_data = pd.read_csv('marvellousTitanicDataSet.csv')

#     print("First 5 entries from loaded dataset")
#     print(titanic_data.head())

#     print("Number of passengers: " + str(len(titanic_data)))

#     # Step 2: Visualize data
#     print("Visualization: Survived and non-survived passengers based on gender")
#     plt.figure()
#     sns.countplot(data=titanic_data, x='Survived', hue='Sex').set_title("Marvellous Infosystem: Survived and non-survived passengers based on gender")
#     plt.show()

#     print("Visualization: Survived and non-survived passengers based on passenger class")
#     plt.figure()
#     sns.countplot(data=titanic_data, x='Survived', hue='Pclass').set_title("Marvellous Infosystem: Survived and non-survived passengers based on passenger class")
#     plt.show()

#     print("Visualization: Survived and non-survived passengers based on age")
#     plt.figure()
#     titanic_data["Age"].plot.hist().set_title("Marvellous Infosystem: Survived and non-survived passengers based on age")
#     plt.show()

#     print("Visualization: Survived and non-survived passengers based on Fare")
#     plt.figure()
#     titanic_data["Fare"].plot.hist().set_title("Marvellous Infosystem: Survived and non-survived passengers based on fare")
#     plt.show()

#     # Step 3: Data Cleaning
#     titanic_data.drop("zero", axis=1, inplace=True)  # Check if this is necessary

#     print("First 5 entries from loaded dataset after removing 'zero' column")
#     print(titanic_data.head(5))

#     print("Values of 'Sex' column")
#     print(pd.get_dummies(titanic_data["Sex"]))

#     print("Values of 'Sex' column after removing one field")
#     sex = pd.get_dummies(titanic_data["Sex"], drop_first=True)
#     print(sex.head(5))

#     print("Values of 'Pclass' column after removing one field")
#     pclass = pd.get_dummies(titanic_data["Pclass"], drop_first=True)
#     print(pclass.head(5))

#     print("Values of dataset after concatenating new columns")
#     titanic_data = pd.concat([titanic_data, sex, pclass], axis=1)
#     print(titanic_data.head(5))

#     print("Values of dataset after removing irrelevant columns")
#     titanic_data.drop(["Sex", "SibSp", "Parch", "Embarked"], axis=1, inplace=True)
#     print(titanic_data.head(5))

#     x = titanic_data.drop("Survived", axis=1)
#     y = titanic_data["Survived"]

#     # Step 4: Data Training
#     xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5)

#     logmodel = LogisticRegression()
#     logmodel.fit(xtrain, ytrain)

#     # Step 5: Data Testing
#     prediction = logmodel.predict(xtest)

#     # Step 6: Calculate Accuracy
#     print("Classification report of Logistic Regression:")
#     print(classification_report(ytest, prediction))

#     print("Confusion matrix of Logistic Regression:")
#     print(confusion_matrix(ytest, prediction))

#     print("Accuracy of Logistic Regression:")
#     print(accuracy_score(ytest, prediction))

# def main():
#     print("----- Marvellous Infosystem by Piyush Khairner ------")
#     print("Supervised machine learning")
#     print("Logistic Regression on Titanic dataset")
#     MarvellousTitanicLogistic()

# if __name__ == "__main__":
#     main()


 import math
 import numpy as np
 import pandas as pd
 import seaborn as sns
 from seaborn import countplot
 import matplotlib.pyplot as plt
 from matplotlib.pyplot import figure, show
 from sklearn.metrics import accuracy_score
 from sklearn.metrics import confusion_matrix
 from sklearn.metrics import classification_report
 from sklearn.model_selection import train_test_split
 from sklearn.linear_model import Logistic Regression

def Marvellous TitanicLogistic():
    
    #step 1: Load data
    titanic_data = pd.read_csv('Marvellous TitanicDataset.csv')


    print("First 5 entries from loaded dataset")
    print(titanic_data.head())


    print("Number of passangers are "+str(len(titanic_data)))


    # Step 2 Analyze data
    print("Visualisation: Survived and non survied passangers")
    figure()
    target = "Survived"


    countplot(data=titanic_data,x=target).set_title("Marvellous Infosystems :Survived and non survied
    passangers")
    show()


    print("Visualisation: Survived and non survied passangers based on Gender")
    figure()
    target = "Survived"


    countplot(data=titanic_data,x=target).set_title("Marvellous Infosystems :Survived and non survied
    passangers")
    show()


    print("Visualisation: Survived and non survied passangers based on Gender")
    figure()
    target "Survived"



    countplot(data-titanic_data,x=target, hue="Sex").set_title("Marvellous Infosystems: Survived and non
    survied passangers based on Gender")
    show()


    print("Visualisation: Survived and non survied passangers based on the Passanger class")
    figure()
    target = "Survived"


    countplot(data-titanic_data,x-target, hue="Pclass").set_title("Marvellous Infosystems: Survived and non
    survied passangers based on the Passanger class")
    show()


    print("Visualisation: Survived and non survied passangers based on Age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Marvellous Infosystems: Survived and non survied passangers based
    on Age")
    show()



    print("Visualisation : Survived and non survied passangers based on the Fare")
    figure()


    titanic_data["Fare"].plot.hist().set_title("Marvellous Infosystems: Survived and non survied passangers based
    on Fare")
    show()


    # Step 3: Data Cleaning
    titanic_data.drop("zero", axis=1, inplace=True)

    print("First 5 entries from loaded dataset after removing zero column")
    print(titanic_data.head(5))

    print("Values of Sex column")
    print(pd.get_dummies (titanic_data["Sex"]))


    print("Values of Sex column after removing one field")
    Sex = pd.get_dummies(titanic_data["Sex"], drop_first = True)
    print(Sex.head(5))


    print("Values of Plass column after removing one field")
    Pclass = pd.get_dummies (titanic_data["Pclass"], drop_first = True)
    print(Pclass.head(5))


    print("Values of data set after concatenating new columns")
    titanic_data = pd.concat([titanic_data, Sex, Pclass],axis=1)
    print(titanic_data.head(5))




    print("Values of data set after removing irrelevent columns")
    titanic_data.drop(["Sex","sibsp", "Parch","Embarked"], axis=1, inplace=True)
    print(titanic_data.head(5))



    y = titanic_data["Survived"]

    x = titanic_data.drop("Survived",axis= 1)

    # Step 4: Data Training
    xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.5)
    logmodel = Logistic Regression()
    logmodel.fit(xtrain, ytrain)


    # Step 4: Data Testing
    prediction=logmodel.predict(xtest)


    #Step 5 Calculate Accuracy
    print("Classification report of Logistic Regression is: ")
    print(classification_report(ytest, prediction))

    print("Confusion Matrix of Logistic Regression is: ")
    print(confusion matrix(ytest, prediction))


    print("Accuracy of Logistic Regression is: ")
    print(accuracy_score (ytest, prediction))

def main():


    print("-- Marvellous Infosystems by Piyush Khairnar-----")

    print("Suervised Machine Learning")


    print("Logistic Regreesion on Titanic data set")


    MarvellousTitanicLogistic()


name_ == "__main__":
main()










