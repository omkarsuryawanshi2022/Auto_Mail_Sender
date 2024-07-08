from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():

    print("Iris flower case study----------")

    data=load_iris()

    #print(shape(iris))
   # print(shape(iris))

   # print(type(data))

    Features = data.data
    Labels=data.target

# terget means = label
    data_train,data_test,target_train,target_test = train_test_split(Features,Labels,test_size=0.5)

    obj=tree.DecisionTreeClassifier()

    obj=obj.fit(data_train,target_train)

    output=obj.predict(data_test)

    Accuracy=accuracy_score(target_test,output)

    print("Accuracy is",Accuracy*100,"%")

if __name__ =="__main__":
    main()
