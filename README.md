# Banknote Classifier

This project is a machine learning model that assesses whether a banknote is forged or not. It is implemented in Python and integrates with XAMPP for database management.

## Overview

The project uses the UCI Banknote Authentication dataset to build and evaluate a classification model. The dataset can be accessed [here](https://archive.ics.uci.edu/dataset/267/banknote+authentication).

### Project Structure

- **access.py**: Imports an SQL dataset from PHPMyAdmin and gets it into a DataFrame. Performs preliminary EDA.
- **stats_tests.py**: 
  - Imports the DataFrame from `access.py`.
  - Performs EDA, including plotting histograms and analyzing target classes before and after balancing.
  - Conducts multicollinearity tests (Bartlett and KMO).
  - Executes PCA, plots a Scree Plot and Biplot.
  - Performs a classification model and prints accuracy and classification report.
  - Saves PCA, Scaler, and Model.
- **test_model.py**: Loads the model and makes predictions on new data.
- **app.py**: Combines all files and streamlines the whole process. Run everything from this file.
- **import_instructions.md**: 
  - Instructions for creating a database and table in XAMPP and PgAdmin4, uploading a dataset, and importing it into Python.
  - How to import the dataset directly from UCI and merge X and Y into a single DataFrame.
- **requirements.txt**: Contains the dependencies for the project, which can be used with Docker.

## Cloning the Repository

To clone this repository and get started with the project, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/banknote-classifier.git
