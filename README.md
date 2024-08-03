# banknote-classifier
machine learning model that assesses whether a banknote is forged or not

This is a full Classification model project done in Python/Xampp.

UCI Dataset: https://archive.ics.uci.edu/dataset/267/banknote+authentication

Files: 

# access.py
1. It imports an SQL dataset from PHPMyAdmin and gets it into a DataFrame
2. Does preliminary EDA
   
# stats_tests.py
1. It imports the DataFrame from Access.py
2. Performs EDA, plots 4 Histograms, plots target classes before and after balancing
3. Performs 2 Multicollinearity Tests (Bartlett and KMO)
4. Performs PCA, Plots a Scree Plot and Biplot
5. Performs a Classification Model
6. Prints Accuracy and Classification Report
7. Saves the PCA, Scaler, and Model

# test_model.py
1. Loads the Model
2. Makes predictions on new data

# app.py
1. Combines all files and streamlines the whole process
2. You can run everything from here

# import instructions
A local file that explains how to:
1.  Create a database, and a table, upload a dataset to XAMPP, and import it into Python.
2.  Create a database, and a table, and then upload a dataset to gAdmin4 and import it into Python.
3.  Import the dataset directly into Python from UCI, but merge X and Y into a single Dataframe.

