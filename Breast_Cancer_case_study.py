# ########################################################
# # Required  python pakages
# ########################################################
# import pandas  as pd
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandamForestClassifier
# from sklearn.ensemble import RandamForestClassifier


# ##############################################################
# #File Path
# ##############################################################
# INPUT_PATH = "breast -cancer-wisconsin.data"
# OUTPUT_PATH = "breast-cancer-wisconsin (2).csv"

# ##############################################################
# #Headers
# ##############################################################
# HEADERS = ["CodeNumber","ClumpThikness","UnifromilityCellSize","UnifromityCellShape",
# "MarginalAdhesion",
# "SigleEpithelialCellSize","BareNucli","BlandChromatin","NormalNucleoli","Mitoses",
# "CancerType"]


# ##############################################################
# #Function Name : read_data
# #Description : Read the data into pandas dataframe
# #inpt : path of CSV file
# #output : Gives the data
# #Author : Omkar Nanaso Suryawanshi
# #Date : 07/07/2024
# ##############################################################
# def read_data(path):
#     data = pd.read_csv(path)
#     return data

# ##############################################################
# #Function Name : get_Headder
# #Description : dataset headers
# #inpt : dataset
# #output : return the headers
# #Author : Omkar Nanaso Suryawanshi
# #Date : 07/07/2024
# ##############################################################
# def get_headers(dataset):
#     return dataset.columns.values

# ##############################################################
# #Function Name : get_Headder
# #Description : dataset headers
# #inpt : dataset
# #output : return the headers
# #Author : Omkar Nanaso Suryawanshi
# #Date : 07/07/2024
# ##############################################################

# def add_headers(dataset,headers):
#     dataset.columns = headers
#     return dataset
# ##############################################################
# #Function Name : get_Headder
# #Description : dataset headers
# #inpt : dataset
# #output : return the headers
# #Author : Omkar Nanaso Suryawanshi
# #Date : 07/07/2024
# ##############################################################

# def data_file_to_csv():
#     #Headers
#     headers = ["CodeNumber","ClumpThikness","UnifromilityCellSize","UnifromityCellShape",
# "MarginalAdhesion",
# "SigleEpithelialCellSize","BareNucli","BlandChromatin","NormalNucleoli","Mitoses",
# "CancerType"]
#     #load the dataset into pandas data frame
#     dataset = read_data(INPUT_PATH)

#     # Add the headers to the loaded dataset
#     dataset = add_headers(dataset,headers)

#     # save  the loaded dataset into csv format
#     dataset.to_csv(OUTPUT_PATH,index=False)
#     print("File Saved")

# ##############################################################
# #Function Name : get_Headder
# #Description : dataset headers
# #inpt : dataset
# #output : return the headers
# #Author : Omkar Nanaso Suryawanshi
# #Date : 07/07/2024
# ##############################################################

# def  split_dataset(dataset,train_percentage,feature_headers,target_header):
#     #Split data set into train  and test dataset

#     train_x,test_x,train_y,test_y = train_test_split(dataset[feature_headers],
#     dataset[target_header],train_size=train_percentage)
#     return train_x,test_x,train_y,test_y

# ##############################################################
# #Function Name : get_Headder
# #Description : dataset headers
# #inpt : dataset
# #output : return the headers
# #Author : Omkar Nanaso Suryawanshi
# #Date : 07/07/2024
# ##############################################################

# def handle_missing_values(dataset,missing_values_header,missing_label):
#     return dataset[dataset[missing_values_header] != missing_label]
# ##############################################################
# #Function Name : get_Headder
# #Description : dataset headers
# #inpt : dataset
# #output : return the headers
# #Author : Omkar Nanaso Suryawanshi
# #Date : 07/07/2024
# ##############################################################
# def random_forest_classifier(feature,target):
#     clf = RandamForestClassifier()
#     clf.fit(feature,target)
#     return clf
# ##############################################################
# #Function Name : get_Headder
# #Description : dataset headers
# #inpt : dataset
# #output : return the headers
# #Author : Omkar Nanaso Suryawanshi
# #Date : 07/07/2024
# ##############################################################
# def dataset_statistics(dataset):
#     print(dataset.describe())
# ##############################################################
# #Function Name : get_Headder
# #Description : dataset headers
# #inpt : dataset
# #output : return the headers
# #Author : Omkar Nanaso Suryawanshi
# #Date : 07/07/2024
# ##############################################################

# def main():
#     #load the csv file into pandas data frame
#     dataset = pd.read_csv(OUTPUT_PATH)

#     #Get Basic statictics of the loaded dataset
#     dataset_statistics(dataset)

#     #filter missing value
#     dataset = handle_missing_values(dataset,HEADERS[6], '?')
#     train_x,test_x,train_y,test_y = split_dataset(dataset,0.7,HEADERS[1:-1], HEADERS[-1])

#     # train and test dataset size details
#     print("Train_x Shape ::",train_x.Shape)
#     print("Train_y Shape ::",train_y.Shape)
#     print("Test_x Shape ::",test_x.Shape)
#     print("Test_y Shape ::",test_y.Shape)

#     #create random forest classifier instance
#     trained_model = random_forest_classifier(train_x,train_y)
#     print("Traned model ::",trained_model)
#     predictions =  trained_model.predict(test_x)

#     for i in range(0,205):
#         print("Actual outcome :: {} and predicted outcome :: {}".format(list(test_y)[i],predictions[i]))
#         print("Train Accuracy ::",accuracy_score(train_y,trained_model.predict(train_x)))
#         print("Test Accuracy ::",accuracy_score(test_y,predictions))
#         print("Confusion Matrix",confusion_matrix(test_y,predictions))

# #################################################
# # Application Starter
# #################################################
# if __name__ == "__main__":
#     main()
    

########################################################
# Required python packages
########################################################
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

##############################################################
# File Paths
##############################################################
INPUT_PATH = "breast-cancer-wisconsin.data"
OUTPUT_PATH = "breast-cancer-wisconsin (2).csv"

##############################################################
# Headers
##############################################################
HEADERS = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape",
           "MarginalAdhesion", "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", 
           "NormalNucleoli", "Mitoses", "CancerType"]

##############################################################
# Function Name: read_data
# Description: Read the data into pandas dataframe
# Input: path of CSV file
# Output: Pandas DataFrame containing the data
# Author: Omkar Nanaso Suryawanshi
# Date: 07/07/2024
##############################################################
def read_data(path):
    data = pd.read_csv(path)
    return data

##############################################################
# Function Name: add_headers
# Description: Add headers to the dataset
# Input: dataset (DataFrame), headers (list)
# Output: DataFrame with headers added
# Author: Omkar Nanaso Suryawanshi
# Date: 07/07/2024
##############################################################
def add_headers(dataset, headers):
    dataset.columns = headers
    return dataset

##############################################################
# Function Name: data_file_to_csv
# Description: Read data, add headers, and save to CSV
# Author: Omkar Nanaso Suryawanshi
# Date: 07/07/2024
##############################################################
def data_file_to_csv():
    # Headers
    headers = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape",
               "MarginalAdhesion", "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", 
               "NormalNucleoli", "Mitoses", "CancerType"]
    
    # Load the dataset into pandas DataFrame
    dataset = read_data(INPUT_PATH)

    # Add headers to the dataset
    dataset = add_headers(dataset, headers)

    # Save the dataset to CSV format
    dataset.to_csv(OUTPUT_PATH, index=False)
    print("File Saved")

##############################################################
# Function Name: split_dataset
# Description: Split dataset into train and test sets
# Input: dataset (DataFrame), train_percentage (float), 
#        feature_headers (list), target_header (str)
# Output: train_x, test_x, train_y, test_y
# Author: Omkar Nanaso Suryawanshi
# Date: 07/07/2024
##############################################################
def split_dataset(dataset, train_percentage, feature_headers, target_header):
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers],
                                                        dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y

##############################################################
# Function Name: handle_missing_values
# Description: Handle missing values in the dataset
# Input: dataset (DataFrame), missing_values_header (str), 
#        missing_label (str)
# Output: DataFrame with missing values handled
# Author: Omkar Nanaso Suryawanshi
# Date: 07/07/2024
##############################################################
def handle_missing_values(dataset, missing_values_header, missing_label):
    return dataset[dataset[missing_values_header] != missing_label]

##############################################################
# Function Name: random_forest_classifier
# Description: Create and train Random Forest classifier
# Input: feature (DataFrame), target (DataFrame)
# Output: trained Random Forest classifier
# Author: Omkar Nanaso Suryawanshi
# Date: 07/07/2024
##############################################################
def random_forest_classifier(feature, target):
    clf = RandomForestClassifier()
    clf.fit(feature, target)
    return clf

##############################################################
# Function Name: dataset_statistics
# Description: Display basic statistics of the dataset
# Input: dataset (DataFrame)
# Author: Omkar Nanaso Suryawanshi
# Date: 07/07/2024
##############################################################
def dataset_statistics(dataset):
    print(dataset.describe())

##############################################################
# Function Name: main
# Description: Main function from where execution starts
# Author: Omkar Nanaso Suryawanshi
# Date: 07/07/2024
##############################################################
def main():
    # Load the CSV file into pandas DataFrame
    dataset = pd.read_csv(OUTPUT_PATH)

    # Display basic statistics of the loaded dataset
    dataset_statistics(dataset)

    # Handle missing values
    dataset = handle_missing_values(dataset, HEADERS[6], '?')

    # Split dataset into train and test sets
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])

    # Print train and test dataset sizes
    print("Train_x Shape :: ", train_x.shape)
    print("Train_y Shape :: ", train_y.shape)
    print("Test_x Shape :: ", test_x.shape)
    print("Test_y Shape :: ", test_y.shape)

    # Create a Random Forest classifier instance
    trained_model = random_forest_classifier(train_x, train_y)
    print("Trained model :: ", trained_model)

    # Make predictions
    predictions = trained_model.predict(test_x)

    # Print accuracy metrics
    print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print("Confusion Matrix ::\n", confusion_matrix(test_y, predictions))

#################################################
# Application Starter
#################################################
if __name__ == "__main__":
    # Execute the main function
    main()




