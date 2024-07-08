# import numpy as np
# import pandas as pd
# from sklearn import preprocessing

# def MarvellousPlayPredictor(data_path):

#     # step 1 data load
#     data = pd.read_csv(data_path,index_col=0)

#     print("Size of the actual dataset",len(data))

#     #step 2 :clean and manipulate data
#     feature_names = ['Whether','Temperature']

#     print("names of features",feature_names)
 
#     Whether = data.Whether
#     Temperature = data.Temperature
#     play = data.play

#     # creating labelencoder

#     le = preprocessing.LabelEncoder()

#     # converting String label into numbers
#     Whether_encoded = le.fit_transform(Whether)
#     print(Whether_encoded)

#     # converting String label into numbers
#     temp_encoded = le.fit_transform(Temperature)
#     label = le.fit_transform(play)
#     print(temp_encoded)

#     # combining weather and temp into single list of tuples
#     feature = list(zip(weather_encoded,temp_encoded))

#     # step 3 : train data
#     model = KNeighborsClassifier(n_neighbors=3)

#     # train the model the tranning sets
#     model.fit(feature,label)

#     # step 4 test the data
#     predictor = model.predict([[0,2]]) # 0:overcast, 2:mild
#     print(predict)

# def main():

#     print("Marvellous infosystem by piyush khairnar---")

#     print("Machine learning Application")

#     print("play predictor application using knighbor Algorithm")

#     MarvellousPlayPredictor("playPredictor.csv")

# if __name__ == "__main__":
#     main()

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

def MarvellousPlayPredictor(data_path):
    # Step 1: Data load
    data = pd.read_csv(data_path)

    print("Size of the actual dataset:", len(data))

    # Step 2: Clean and manipulate data
    feature_names = ['Whether', 'Temperature']

    print("Names of features:", feature_names)

    # Correcting column names to match the data
    try:
        Whether = data['Whether']
        Temperature = data['Temperature']
        Play = data['Play']
    except KeyError as e:
        print(f"KeyError: {e}. Check if the column names are correct in your CSV file.")
        return

    # Creating label encoder
    le = preprocessing.LabelEncoder()

    # Converting string labels into numbers
    Whether_encoded = le.fit_transform(Whether)
    print("Encoded Whether:", Whether_encoded)

    # Converting string labels into numbers
    Temp_encoded = le.fit_transform(Temperature)
    Label = le.fit_transform(Play)
    print("Encoded Temperature:", Temp_encoded)

    # Combining Weather and Temperature into single list of tuples
    feature = list(zip(Whether_encoded, Temp_encoded))

    # Step 3: Train data
    model = KNeighborsClassifier(n_neighbors=3)

    # Train the model using the training sets
    model.fit(feature, Label)

    # Step 4: Test the data
    predictor = model.predict([[0, 2]])  # 0: Overcast, 2: Mild
    print("Prediction:", predictor)

def main():
    print("Marvellous Infosystem by Piyush Khairnar")
    print("Machine Learning Application")
    print("Play predictor application using K-Nearest Neighbors Algorithm")
    MarvellousPlayPredictor("playPredictor.csv")

if __name__ == "__main__":
    main()
