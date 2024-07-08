from sklearn import tree
# Rough=1
#smooth=0

# tennis 1
# cricket 2
def MarvellousClassifier(weight,surface):

    # feature Encoding
    Feature=[[35,1], [47,1], [90,0], [48,1],
     [90,0], [35,1], [92,0], [35,1],[35,1],
      [35,1]]

    Labels=[1,1,2,1,2,1,2,1,1]

    #train the models
    obj=tree.DecisionTreeClassifier()

    # train the mode

    obj=obj.fit(Features,Labels)


    ret=obj.predict([[weight,surface]])
    if ret ==1:
        print("your object looks like tennis boll")
    else:
        print("object like cricket boll")




def main():

    print("Ball  type classification case study")

    print("please enter the informatiomobout the object that you want to  test")

    print("please enter the weight of your object in gram")
    no=int(input())

    print("please mention the type of surface rough/smooth")
    data=input()

    if data.lower=="rough":
        data=1
    elif data.lower=="Smooth":
        data=0
    else:
        print("invalid type of surface ")
        exit()

    MarvellousClassifier(no,data)



if __name__ =="__main__":
    main()

    #data size=15
