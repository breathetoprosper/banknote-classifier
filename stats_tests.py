"""
1 indicates that the banknote is genuine.
0 indicates that the banknote is counterfeit.
"""
from access import get_dataframe # from the module access we are importing the function get_dataframe that has the dataframe
import pandas as pd # this one is to be able to create and use dataframes
import matplotlib.pyplot as plt  # this one is to show graphical plots
import matplotlib.colors as mcolors # to map the colors of the scatter dots in teh biplot
import matplotlib.patches as patches # to outline the vectors in a different color in the biplot
import numpy as np # this one is to array operations
import joblib # this one is for saving and loading Python objects, especially large data structures like machine learning models.
from scipy.stats import bartlett # this one is to perform a multicolinearity tests.
from factor_analyzer.factor_analyzer import calculate_kmo # same for this one is to perform a multicolinearity tests.
from sklearn.decomposition import PCA # this one is to performe principal component analysis
from sklearn.preprocessing import StandardScaler # this one is to standardize variables.
from sklearn.model_selection import train_test_split # this one is to split training test set.
from sklearn.ensemble import RandomForestClassifier # this will be our machine learning classifier algorithm.
from sklearn.metrics import accuracy_score, classification_report # this is for us to see the results
from sklearn.utils import resample # this is required for balancing data, as we are going to resample.

COLUMNS_OF_INTEREST = ['variance', 'skewness', 'kurtosis', 'entropy'] # these are our features
TARGET = 'class' # this is the target column.

# here we do our first analysis. Getting the dataset info and also passing in a dynamic name so that
# later on we can see the results for the reduced version
def print_dataset_info(df, name):
    print(f"\n{name} Dataset Summary Information:")
    print(df.info())
    print(f"\n{name} Dataset First Columns:")
    print(df.head())
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    print(f"\n{name} Dataset Summary Descriptive Statistics:")
    print(df.describe(include='all'))   

# here we are going to do 2 things:
# 1. grammar from teh curtosis->kurtosis and 
# 2. randomly shuffle the dataset. We will need this later on when we get a chunk from it.
def preprocess_dataframe(df):
    # Rename columns and shuffle dataframe
    df.rename(columns={'curtosis': 'kurtosis'}, inplace=True)
    df_shuffled = df.sample(frac=1, random_state=37).reset_index(drop=True)
    return df_shuffled

# here we get the dataframe from access.py and get a chunk from it of 20% to use as valudation in the end.
# we are using 80% of it to split into training/test set.
def split_dataframe(df_shuffled, validation_fraction=0.2):
    """
    Split the shuffled DataFrame into validation and training/testing sets.
    
    Parameters:
    - df_shuffled: The shuffled DataFrame to be split.
    - validation_fraction: The fraction of the DataFrame to be used as the validation set.
    
    Returns:
    - df_val: The validation set DataFrame.
    - df_tt: The training/testing set DataFrame.
    
    Code Explanation:
    iloc: This is an indexer for pandas DataFrames that allows you to 
    select rows and columns by their numerical indexes instead of values like labels or names.
    So it allows you to select rows and columns by their integer positions when that is precisely
    what is needed.
    
    [:split_index]: This is slicing syntax. 
    It selects all rows from the beginning of the DataFrame up to, 
    but not including, the split_index
    
    reset_index(): This method resets the index of the DataFrame.
    drop=True: This argument ensures that the old index 
    is not added as a new column in the DataFrame. 
    It simply resets the index to the default integer values (0, 1, 2, ...).
    
    so as a result both dataframes have fresh indexes starting at 0.
    

    """
    # Calculate the index to split the DataFrame. int truncates 20% * lenght of df_shuffled
    # say 1004.1 is 1004 here.
    split_index = int(validation_fraction * len(df_shuffled))
    
    # Split the DataFrame into validation and training/testing sets
    df_val = df_shuffled.iloc[:split_index].reset_index(drop=True)
    df_tt = df_shuffled.iloc[split_index:].reset_index(drop=True)

   # Return the split DataFrames
    return df_val, df_tt

# here we return the validation dataset(the 20% one so that we can then use it in the end.)
def get_validation_data():
    """
    Retrieve and preprocess the validation data.

    This function fetches a DataFrame, preprocesses it, splits it into validation
    and training/testing sets, and returns the validation set.

    Returns:
    - df_val: The DataFrame used for validation.
    
    Code Explanation:
    
    """
    # Retrieve the DataFrame
    df = get_dataframe()

    # Preprocess the DataFrame and shuffle it.
    df_shuffled = preprocess_dataframe(df)
    
    # Split the DataFrame into validation and training/testing sets.
    # here we are only interested in dv_val hence we skip df_tt
    df_val, _ = split_dataframe(df_shuffled)

    # Extract the feature columns and the target labels
    X_val = df_val[COLUMNS_OF_INTEREST]
    y_val = df_val[TARGET]

    # Return the validation set
    return df_val

def plot_data(df, title):
    """
    Plot histograms for the numerical columns of the DataFrame, excluding 'class' and 'id' columns.
    The function plots histograms for up to 4 numerical columns, arranged in a grid layout.
    
    Parameters:
    - df: The DataFrame containing the data to be plotted.
    """
    # Histograms
    # Exclude 'class' and 'id' columns from the plotting
    # we are saying: if the col is not 'class' or 'id', add it to columns_to_plot
    columns_to_plot = [col for col in df.columns if col not in ['class', 'id']]
    
   # Adjust the figure size to fit the number of plots
    columns_to_plot = columns_to_plot[:4]
    num_columns = min(len(columns_to_plot), 2)
    num_rows = (len(columns_to_plot) + num_columns - 1) // num_columns # (4+2-1) //2 = 5 // 2 = 4 

    # We use this to adjust the figure size for a smaller/larger window
    plt.figure(figsize=(num_columns * 4, num_rows * 4)) # Set figure size based on the number of columns and rows
    
    for i, column in enumerate(columns_to_plot):
        plt.subplot(num_rows, num_columns, i + 1)  # Create a subplot for each column
        data = df[column]  # Extract the data for the current column
        n = len(data)  # Number of data points
        bins = int(np.sqrt(n))  # Here se use sqrt(n) to calculate number of bins as sqrt of number of bins
        if bins < 1:  # Ensure at least one bin is used
            bins = 1
        data.hist(bins=bins, edgecolor='black') # Plot the histogram with the calculated number of bins
        plt.title(column)  # Set the title of the subplot to the column name
        plt.xlabel('Value')  # Label for the x-axis
        plt.ylabel('Frequency')  # Label for the y-axis
    
    plt.suptitle(f'Histograms of the {title} Set', fontsize=20)  # Set the main title for the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # [left, bottom, right, top]
    """
    Adjust layout to make room for the main title.
    - left=0: Starts from the left edge of the figure.
    - bottom=0: Starts from the bottom edge of the figure.
    - right=1: Extends to the full width of the figure.
    - top=0.96: Leaves 4% at the top for the title.
    """
    plt.show()

def multicolinearity_tests(df):
    """
    Perform multicollinearity tests on the DataFrame to assess 
    the suitability of the data for factor analysis.
    
    Parameters:
    - df: The DataFrame containing the data for analysis.
    """
    # Select only the columns of interest from the DataFrame
    df_selected = df[COLUMNS_OF_INTEREST]

    # Convert all columns to numeric values. Any non-numeric values are set to NaN.
    # The `errors='coerce'` parameter handles the conversion of non-numeric values to NaN.
    df_selected = df_selected.apply(pd.to_numeric, errors='coerce').dropna()
    
    #This gets the number of rows in df_selected. 
    # if less than 2 states that we don't have enough data.
    if df_selected.shape[0] < 2:
        print("Not enough data points for KMO test after dropping NaNs.")
        return
    
    # Perform Bartlett's Sphericity Test to check if the correlation matrix 
    # is significantly different from an identity matrix
    # The * operator unpacks the list of columns into separate arguments for the test
    # it returns the test statistic and the p_value result of the test.
    stat, p_value = bartlett(*[df_selected[col] for col in COLUMNS_OF_INTEREST])

    # Print the results of Bartlett's test
    print("Bartlett's Sphericity Test:")
    print("Bartlett's test statistic:", stat)
    print("Bartlett's p-value:", p_value)

    # Set the significance level
    alpha = 0.05
    print("Significance level:", alpha)
    
    # Interpretation the p-value from Bartlett's test
    if p_value < alpha:
        print(f"p-value ({p_value:.4f}) < alpha ({alpha})")
        print("Favor H1 - at least 2 variances are different")
        print("Decision: Reject the null hypothesis of equal variances.")
        print("Variances are heterogeneous.")
        print("Implication: The correlation matrix is significantly different from an identity matrix.")
        print("This suggests that there are meaningful correlations among the variables.")
        print("You may proceed with PCA.")
    else:
        print(f"p-value ({p_value:.4f}) >= alpha ({alpha})")
        print("Favor H0 - all variances are the sames")
        print("Decision: Fail to reject the null hypothesis of equal variances.")
        print("Variances are homogeneous.")
        print("Implication: There is insufficient evidence to say the correlation matrix is different from an identity matrix.")
        print("This suggests that the variables may not be significantly correlated.")
        print("PCA may not be appropriate, consider other methods or further investigation.")
    
    # Perform Kaiser-Meyer-Olkin (KMO) Test to assess the suitability of the data for factor analysis
    print("\nKaiser-Meyer-Olkin (KMO) Test:")
    kmo_all, kmo_model = calculate_kmo(df_selected)
    
    # Create a DataFrame to display KMO statistics for each variable
    kmo_df = pd.DataFrame({
        'Variable': COLUMNS_OF_INTEREST,
        'KMO Statistic': kmo_all
    })
    
    # Print KMO decision rule
    print("KMO decision rule:")
    print("0.8 - 1.0: Excellent")
    print("0.7 - 0.79: Good")
    print("0.6 - 0.69: Mediocre")
    print("0.5 - 0.59: Poor")
    print("Below 0.5: Unacceptable\n")

    # Print KMO statistic for each variable
    print("KMO statistic for each variable:")
    for variable, kmo_value in zip(kmo_df['Variable'], kmo_df['KMO Statistic']):
        print(f"{variable}: {kmo_value}")

    # Print overall KMO statistic
    print("\nKMO statistic (overall):", kmo_model)

    # Print a dynamic conclusion based on overall KMO statistic
    if kmo_model >= 0.8:
        conclusion = "Excellent - Data is suitable for factor analysis."
    elif kmo_model >= 0.7:
        conclusion = "Good - Data is likely suitable for factor analysis."
    elif kmo_model >= 0.6:
        conclusion = "Mediocre - Data may be suitable for factor analysis, but results should be interpreted with caution."
    elif kmo_model >= 0.5:
        conclusion = "Poor - Data is not very suitable for factor analysis. Consider revising the data or the analysis method."
    else:
        conclusion = "Unacceptable - Data is not suitable for factor analysis. Reconsider the use of factor analysis."

    # Print the conclusion based on KMO results
    print("KMO Conclusion:")
    print(conclusion, "\n")

# to be able to do PCA we need to standardize the data first.
# In Python, None is a special constant that represents the absence of a value or a null value. 
# It is often used to indicate that a variable has no value assigned to it.
# The default value of X_test is None if you don't provide it.
# Since it is provided, 
# The transformation of X_test only occurs
def standardize_data(X_train, X_test=None):
    """
    Standardize the features of the training and test datasets.
    
    Parameters:
    - X_train: The training feature matrix to be standardized.
    - X_test: The test feature matrix to be standardized (optional).
    
    Returns:
    - X_train_scaled: The standardized training feature matrix.
    - X_test_scaled: The standardized test feature matrix (if X_test is provided).
    - scaler: The StandardScaler object used for scaling.
    """
    # Initialize the StandardScaler to standardize the data
    scaler = StandardScaler()

    # Fit the scaler to the training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform the test data using the same scaler
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    # return the train and test features X and the scaler object
    return X_train_scaled, X_test_scaled, scaler

# here we perform the PCA
def perform_pca(X_train_scaled, X_test_scaled=None, n_components=None):
    """
    Perform Principal Component Analysis (PCA) on the standardized data.
    
    Parameters:
    - X_train_scaled: The standardized training feature matrix.
    - X_test_scaled: The standardized test feature matrix (optional).
    - n_components: The number of principal components to retain (optional).
    
    Returns:
    - X_train_pca: The training data transformed into principal components.
    - X_test_pca: The test data transformed into principal components (if X_test_scaled is provided).
    - pca: The PCA object used for transformation.
    """
    # Initialize PCA with the specified number of components
    pca = PCA(n_components=n_components)

    # Fit PCA to the training data and transform it into principal components
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Transform the test data into principal components using the same PCA
    X_test_pca = pca.transform(X_test_scaled) if X_test_scaled is not None else None
    
    # return the train and test features X after pca and the pca object
    return X_train_pca, X_test_pca, pca

# here we plot the biplot.
def plot_pca_biplot(X_pca, pca, feature_names, y_train):
    """
    Plot a PCA biplot to visualize the principal components and feature loadings.
    
    Parameters:
    - X_pca: The data transformed into principal components.
    - pca: The PCA object used for dimensionality reduction.
    - feature_names: The names of the features used in the PCA.
    
    Biplot Explanation:
    The length of an arrow indicates the importance of the variable 
    in explaining the variation in the data. 
    Longer arrows represent variables with higher loadings(weights) on the principal components.
    The angle between two arrows indicates the correlation between the corresponding variables.s
    Small angle: Positive correlation
    Large angle: Negative correlation
    Angle close to 90 degrees: Low correlation
    
    Code Explanation:
    c=y_train provides the class labels for each data point.
    cmap maps these labels to colors defined in the colors list.
    
    """
    
    # the oder of colots matter so skyblue goes to 0 and salmon to 1.
    colors = ['skyblue', 'salmon']
    cmap = mcolors.ListedColormap(colors)    
    # Create a new figure for the biplot
    plt.figure(figsize=(10, 7))
    # Scatter plot of the first two principal components
    #plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, edgecolors='k', color='#ccffff')
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap=cmap, alpha=0.7, edgecolors='k')
    plt.xlabel('Principal Component 1')  # Label for the x-axis
    plt.ylabel('Principal Component 2')  # Label for the y-axis
    plt.title('PCA Biplot')  # Title of the plot
    
    # Calculate feature loadings for the biplot
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    """
    pca.components_ is an array that contains the principal axes in feature space, 
    representing the directions of maximum variance in the data. 
    Each row corresponds to a principal component, 
    and each column corresponds to a feature. 
    The shape of pca.components_ is (n_components, n_features).
    
    pca.explained_variance_: This is an array of the amount of variance explained 
    by each of the selected components. 
    The shape of pca.explained_variance_ is (n_components,).
    
    T is for transposing pca.components_ 
    It changes its shape from (n_components, n_features) to (n_features, n_components). 
    This makes each column correspond to a principal component 
    and each row correspond to a feature.
    
    np.sqrt(pca.explained_variance_): This calculates the square root of each element 
    in the pca.explained_variance_ array. 
    The result is an array of the standard deviations of the principal components, 
    since the variance is the square of the standard deviation. 
    The shape of np.sqrt(pca.explained_variance_) is (n_components,).
    
    pca.components_.T * np.sqrt(pca.explained_variance_)
    performs an element-wise multiplication of the transposed components matrix 
    and the standard deviations of the principal components. 
    This operation scales each component by the square root 
    of its corresponding explained variance, resulting in the loadings matrix.
    
    As for the plots:
    enumerate(feature_names) returns both the index (i) 
    and the value (feature) for each item in feature_names
    
    head_width and head_length control the size of the arrowhead.
    fc (face color) and ec (edge color) set the color of the arrowhead 
    and the outline of the arrow, respectively. 

    linewidth sets the width of the arrow line. 
    It is set to 4, making the arrow line relatively thick.
    color specifies the color of the arrow line. 
    Here, it is set to 'black'. 
    However, since fc and ec are set to 'red', 
    the arrow line color specified by color might be ignored 
    in this context or it might affect the color of the arrow if fc and ec do not fully apply.
    
    """
    
    # Plot arrows representing the feature loadings with its textual names
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                  head_width=0.05, head_length=0.1,
                  fc='red', ec='red',
                  linewidth=4, color='red')
        # Annotate each feature with its name
        plt.text(loadings[i, 0] * 1.2, loadings[i, 1] * 1.2,
                 feature, color='black',
                 ha='center', va='center',
                 fontsize=16, fontweight='bold')

    # Create a legend with specific colors for the labels
    plt.scatter([], [], color='salmon', label='1 - Genuine')  # Empty scatter plot for legend
    plt.scatter([], [], color='skyblue', label='0 - Fake')  # Empty scatter plot for legend
    plt.legend(title='Class')  # Add legend with a title
    
    plt.grid(True)  # Add grid lines for better readability
    plt.show() # Display the plot

# here we plot the scree to see the Principal Components
def plot_scree(explained_variance_ratio):
    """
    Plot a scree plot to visualize the explained variance ratio of each principal component.
    
    Parameters:
    - explained_variance_ratio: The proportion of variance explained by each principal component.
    """
    # Create a new figure for the scree plot
    plt.figure(figsize=(8, 5))

    # X values represent the principal component indices
    x_values = range(1, len(explained_variance_ratio) + 1)

    # Plot the explained variance ratio for each principal component
    plt.plot(x_values, explained_variance_ratio, 'bo-', markersize=10)
    plt.title('Scree Plot')  # Title of the plot
    plt.xlabel('Principal Component')  # Label for the x-axis
    plt.ylabel('Explained Variance Ratio')  # Label for the y-axis
    plt.xticks(x_values)  # Set x-axis ticks to match the principal components
    plt.grid(True)  # Add grid lines for better readability

   # Annotate each point with its explained variance ratio 
    offset = 0.02  # Offset to prevent text from overlapping with the points
    for x, y in zip(x_values, explained_variance_ratio):
        plt.text(x, y + offset, f'{y:.2f}', fontsize=9, fontweight='bold', ha='center', va='bottom')
    
    plt.show()  # Display the plot

def pca_analysis(X_train, X_test=None, X = None, y_train=None, n_components=0.90):
    """
    Perform PCA on the training and test data, determine the optimal number of components,
    and evaluate the PCA results. Additionally, generate and plot a PCA biplot.
    
    Parameters:
    - X_train: The training feature matrix.
    - X_test: The test feature matrix (optional).
    - n_components: The threshold for cumulative explained variance to determine the number of components.
    
    Returns:
    - X_train_pca: The training data transformed into principal components.
    - X_test_pca: The test data transformed into principal components (if provided).
    - pca: The PCA object used for dimensionality reduction.
    """
    # Standardize the training (and test, if provided) data
    X_train_scaled, X_test_scaled, scaler = standardize_data(X_train, X_test)
    
    # Perform initial PCA to get explained variance ratios
    _, _, pca_full = perform_pca(X_train_scaled)
    explained_variance_ratio = pca_full.explained_variance_ratio_
    
    # Calculate cumulative explained variance to determine the number of components
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    num_components = np.argmax(cumulative_explained_variance >= n_components) + 1
    
    # Perform PCA with determined number of components
    X_train_pca, X_test_pca, pca = perform_pca(X_train_scaled, X_test_scaled, n_components=num_components)

    # Print results: number of chosen components and explained variance ratios
    print(f"Number of Chosen Components: {num_components}\n")
    print("Explained variance ratio:")
    print(explained_variance_ratio, "\n")
    print("Cumulative explained variance ratio:")
    print(cumulative_explained_variance)
    
    # Plot the scree plot to visualize the explained variance ratio
    plot_scree(explained_variance_ratio)
    
    # Save the scaler and PCA model for future use
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(pca, 'pca.pkl')
    
    # Plot the PCA biplot for the training data
    plot_pca_biplot(X_train_pca, pca, COLUMNS_OF_INTEREST, y_train)
    
    return X_train_pca, X_test_pca, pca

# here we train the mdoe using the Random Forest Classifier
def train_and_evaluate_model(X_train_pca, X_test_pca, y_train, y_test):
    """
    Train a Random Forest Classifier on the PCA-transformed training data,
    evaluate it on the PCA-transformed test data, and print the results.
    
    Parameters:
    - X_train_pca: The training data transformed into principal components.
    - X_test_pca: The test data transformed into principal components.
    - y_train: The true labels for the training data.
    - y_test: The true labels for the test data.
    
    Returns:
    - accuracy: The accuracy score of the model on the test data.
    - classification: The classification report of the model's performance.
    """
    # Initialize and train the Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train_pca, y_train)

    # Predict the labels for the test data
    y_pred = model.predict(X_test_pca)

    # Calculate accuracy and generate classification report
    accuracy = accuracy_score(y_test, y_pred)
    classification = classification_report(y_test, y_pred)

    # Save the trained model to a file
    joblib.dump(model, 'random_forest_model.pkl')
    
    # Print the model's performance
    print("->Chosen Model: Random Forests Model")
    print(f"Model accuracy with Training/Test Data Split: {accuracy:.4f}")
    print("Classification report:\n", classification)
    
    return accuracy, classification

# this is a function to show the class distribution of 0's and 1's
def class_distribution(df, name):
    """
    Print and plot the class distribution in a given dataset.
    
    Parameters:
    - df: The DataFrame containing the dataset.
    - name: A string representing the name of the dataset (e.g., 'training', 'test').
    
    Returns:
    - df: The original DataFrame (unchanged).
    """
    # Calculate and print the class distribution
    class_distribution = df[TARGET].value_counts()
    print(f"Class distribution in the {name} dataset:\n", class_distribution, "\n")

    # Plot the class distribution as a bar chart  
    plt.figure(figsize=(8, 5))
    bars = class_distribution.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title(f"Class Distribution in {name} dataset")
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(ticks=[0, 1], labels=['Class 0 - Fake', 'Class 1 - Genuine'], rotation=0)
    plt.grid(axis='y')
    
    # Annotate the bar plot with the class frequencies
    for bar in bars.patches:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{height}', ha='center', va='bottom')
    
    plt.show()
    
    return df

# this is a function to balance the data. 
# Since we have lower (minority) represented class, we wil resample it to get both to 50/50 level.
def balance_data(df):
    """
    Balance the dataset by upsampling the minority class to match the majority class.
    
    Parameters:
    - df: The DataFrame containing the dataset to be balanced.
    
    Returns:
    - df_balanced: A DataFrame with balanced class distribution.
    
    Code Explanation:
    class_counts = df[TARGET].value_counts() # class_counts is a pandas series(one dim-array) that keeps
    the frequencies of df[target] col. So that we can then get the max and min number of occurrences
    wr do here.
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    These are now 2 dataframes where one holds the vector with highest Target Occurrences
    ant the other one the least. We don't know if it is 0 or 1, so this is dynamically achieved.
    df_majority = df[df[TARGET] == majority_class]
    df_minority = df[df[TARGET] == minority_class]
    """
    # Count the number of samples in each class
    class_counts = df[TARGET].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    # Separate majority and minority class samples
    df_majority = df[df[TARGET] == majority_class]
    df_minority = df[df[TARGET] == minority_class]
    
    # Upsample the minority class to match the number of majority class samples
    df_minority_upsampled = resample(df_minority, 
                                     replace=True, 
                                     n_samples=len(df_majority), 
                                     random_state=42)
    
    # Combine the majority class samples with the upsampled minority class samples
    # in a new dataframe df_balanced
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    # and then return it
    return df_balanced

# this is just a function to print the ** over and under the EDA and so on to calculate and print it automatically.
def print_section(title, border_char='*'):
    """
    Print a section title with a border for better visibility in the output.
    It automatically adjusts the number of * to the text so that we get:
    *******
    * EDA *
    *******
    Parameters:
    - title: The title of the section to print.
    - border_char: The character used to create the border (default is '*').
    """
    # Calculate the length of the border based on the title length
    border_length = len(title) + 4  # 2 spaces on each side
    border = border_char * border_length

    # Print the section title with border
    print(f"\n{border}")
    print(f"{border_char} {title} {border_char}")
    print(f"{border}")

# here we call out the main functions
def main():
    """
    Main function to execute the data processing, transformation, and modeling pipeline.
    """
    df = get_dataframe()
    
    # EDA
    print_section("EXPLORATORY DATA ANALYSIS (EDA)")
    print(print_dataset_info(df, 'Original'))
    
    # Shuffle and split the dataset
    df_shuffled = preprocess_dataframe(df)
    df_val, df_tt = split_dataframe(df_shuffled)
    print(print_dataset_info(df_tt, 'Training/Testing'))
    # also print and plot the validation dataset so we can see the comparison of the descriptive statistics
    # and histograms with the training/test dataset
    print(print_dataset_info(df_val, 'Validation'))
    plot_data(df_tt, 'Training/Test') # Training/Test Dataset
    plot_data(df_val, 'Validation') # Validation Dataset
    
    # Data Transformation
    print_section("DATA TRANSFORMATION")
    print("\n->Multicolinearity Tests: Bartlett and KMO\n")
    multicolinearity_tests(df_tt)
    
    print("->Data Balancing:\n")
    _ = class_distribution(df_tt, "original")
    df_balanced = balance_data(df_tt)
    _ = class_distribution(df_balanced, "balanced")

    X = df_balanced[COLUMNS_OF_INTEREST]
    y = df_balanced[TARGET]
    
    print("->Principle Components Analysis (PCA):\n")   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_pca, X_test_pca, _ = pca_analysis(X_train, X_test, X, y_train)
    
    # Data Modelling
    print_section("DATA MODELLING (MODEL BUILDING)")
    _, _ = train_and_evaluate_model(X_train_pca, X_test_pca, y_train, y_test)

# this if-statement allows us to run our module directly.
# if this module is executed as the main program, 
# the main function will be executed.
if __name__ == "__main__":
    main()