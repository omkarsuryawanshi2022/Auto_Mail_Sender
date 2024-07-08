from sklearn import tree
# Rough=1
#smooth=0

# tennis 1
# cricket 2
def MarvellousClassifier():

    # feature Encoding
    Feature=[[35,1], [47,1], [90,0], [48,1],
     [90,0], [35,1], [92,0], [35,1],[35,1],
      [35,1]]

    Labels=[1,1,2,1,2,1,2,1,1]

    #train the models
    obj=tree.DecisionTreeClassifier()

    # train the mode

    obj=obj.fit(Features,Labels)


    ret=obj.predict([[96,0]])
    if ret ==1:
        print("your object looks loke tennis boll")
    else:
        print("object criket boll")




def main():

    print("Ball  type classification case study")

    MarvellousClassifier



if __name__ =="__main__":
    main()

    #data size=15
