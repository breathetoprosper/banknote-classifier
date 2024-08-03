from access import get_dataframe # from the module access we are importing the function get_dataframe that has the dataframe
import pandas as pd # this one is to be able to create and use dataframes
import matplotlib.pyplot as plt  # this one is to show graphical plots
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
    print(f"\n{name} Dataset Summary Descriptive Stastistics:")
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
    # Split the shuffled dataframe into validation and training/testing sets
    split_index = int(validation_fraction * len(df_shuffled))
    df_val = df_shuffled.iloc[:split_index].reset_index(drop=True)
    df_tt = df_shuffled.iloc[split_index:].reset_index(drop=True)
    return df_val, df_tt

# here we return the validation dataset(the 20% one so that we can then use it in the end.)
def get_validation_data():
    df = get_dataframe()
    df_shuffled = preprocess_dataframe(df)
    df_val, _ = split_dataframe(df_shuffled)
    return df_val

# here we plot initial histograms for the remaining 80% chunk we will be working on as training/test set.  
def plot_data(df):
    if df is not None and not df.empty:
        # Histograms
        # Exclude 'class' and 'id' columns from the plotting
        columns_to_plot = [col for col in df.columns if col not in ['class', 'id']]
        
        # Limit to a maximum of 4 columns for plotting
        columns_to_plot = columns_to_plot[:4]
        num_columns = min(len(columns_to_plot), 2)
        num_rows = (len(columns_to_plot) + num_columns - 1) // num_columns

        # We use this to adjust the figure size for a smaller/larger window
        plt.figure(figsize=(num_columns * 4, num_rows * 4))  # Smaller width and height
        
        for i, column in enumerate(columns_to_plot):
            plt.subplot(num_rows, num_columns, i + 1)
            data = df[column]
            n = len(data)
            bins = int(np.sqrt(n))  # Here se use sqrt(n) to calculate number of bins as sqrt of number of bins
            if bins < 1:  # Ensure at least one bin is used
                bins = 1
            data.hist(bins=bins, edgecolor='black')
            plt.title(column)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
        
        plt.suptitle('Histograms', fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    else:
        print("Failed to retrieve data or DataFrame is empty.")

def multicolinearity_tests(df):
    df_selected = df[COLUMNS_OF_INTEREST]
    #This line converts all columns in df_selected to numeric values. 
    #The errors='coerce' parameter ensures that any values that cannot be converted to numeric (e.g., strings) are set to NaN (Not a Number).
    df_selected = df_selected.apply(pd.to_numeric, errors='coerce').dropna()
    
    #This gets the number of rows in df_selected. if less than 2 states that we don't have enough data.
    if df_selected.shape[0] < 2:
        print("Not enough data points for KMO test after dropping NaNs.")
        return
    
    # bartlett's test runs through the datafram, returns the test statistic and the associated p-value.
    # The * operator unpacks the list into separate arguments.
    
    stat, p_value = bartlett(*[df_selected[col] for col in COLUMNS_OF_INTEREST])

    print("Bartlett's Sphericity Test:")
    print("Bartlett's test statistic:", stat)
    print("Bartlett's p-value:", p_value)
    alpha = 0.05
    print("Significance level:", alpha)
    
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
    
    print("\nKaiser-Meyer-Olkin (KMO) Test:")
    kmo_all, kmo_model = calculate_kmo(df_selected)
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

    # Dynamic conclusion based on overall KMO statistic
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

    print("KMO Conclusion:")
    print(conclusion, "\n")

# to be able to do PCA we need to standardize the data first.
def standardize_data(X_train, X_test=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return X_train_scaled, X_test_scaled, scaler

# here we perform the PCA
def perform_pca(X_train_scaled, X_test_scaled=None, n_components=None):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled) if X_test_scaled is not None else None
    return X_train_pca, X_test_pca, pca

# here we plot the biplot.
def plot_pca_biplot(X_pca, pca, feature_names):
    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, edgecolors='k', color='#ccffff')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Biplot')
    
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                  head_width=0.05, head_length=0.1,
                  fc='red', ec='red',
                  linewidth=4, color='red')
        plt.text(loadings[i, 0] * 1.2, loadings[i, 1] * 1.2,
                 feature, color='black',
                 ha='center', va='center',
                 fontsize=16, fontweight='bold')
    
    plt.grid(True)
    plt.show()

# here we plot the scree to see the Principal Components
def plot_scree(explained_variance_ratio):
    plt.figure(figsize=(8, 5))
    x_values = range(1, len(explained_variance_ratio) + 1)
    plt.plot(x_values, explained_variance_ratio, 'bo-', markersize=10)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(x_values)
    plt.grid(True)
    
    offset = 0.02
    for x, y in zip(x_values, explained_variance_ratio):
        plt.text(x, y + offset, f'{y:.2f}', fontsize=9, fontweight='bold', ha='center', va='bottom')
    
    plt.show()

def pca_biplot(X_train):
    X_train_scaled, _, _ = standardize_data(X_train)
    X_pca, _, pca = perform_pca(X_train_scaled, n_components=2)
    plot_pca_biplot(X_pca, pca, COLUMNS_OF_INTEREST)

# here we see the results of the PCA test
def pca_test(X_train, X_test, n_components=0.90):
    X_train_scaled, X_test_scaled, scaler = standardize_data(X_train, X_test)
    
    # Determine number of components
    _, _, pca_full = perform_pca(X_train_scaled)
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    num_components = np.argmax(cumulative_explained_variance >= n_components) + 1
    
    # Perform PCA with determined number of components
    X_train_pca, X_test_pca, pca = perform_pca(X_train_scaled, X_test_scaled, n_components=num_components)
    
    print(f"Number of Chosen Components: {num_components}\n")
    print("Explained variance ratio:")
    print(explained_variance_ratio, "\n")
    print("Cumulative explained variance ratio:")
    print(cumulative_explained_variance)
    
    plot_scree(explained_variance_ratio)
    
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(pca, 'pca.pkl')
    
    pca_biplot(X_train)
    
    return X_train_pca, X_test_pca, pca

# here we train the mdoe using the Random Forest Classifier
def train_and_evaluate_model(X_train_pca, X_test_pca, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    classification = classification_report(y_test, y_pred)
    joblib.dump(model, 'random_forest_model.pkl')
    
    print("->Chosen Model: Random Forests Model")
    print(f"Model accuracy with Training/Test Data Split: {accuracy:.2f}")
    print("Classification report:\n", classification)
    
    return accuracy, classification

# this is a function to show the class distribution of 0's and 1's
def class_distribution(df, name):
    class_distribution = df[TARGET].value_counts()
    print(f"Class distribution in the {name} dataset:\n", class_distribution, "\n")
    
    plt.figure(figsize=(8, 5))
    bars = class_distribution.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title(f"Class Distribution in {name} dataset")
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'], rotation=0)
    plt.grid(axis='y')
    
    for bar in bars.patches:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{height}', ha='center', va='bottom')
    
    plt.show()
    
    return df

# this is a function to balance the data. 
# Since we have lower (minority) represented class, we wil resample it to get both to 50/50 level.
def balance_data(df):
    class_counts = df[TARGET].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    df_majority = df[df[TARGET] == majority_class]
    df_minority = df[df[TARGET] == minority_class]
    
    df_minority_upsampled = resample(df_minority, 
                                     replace=True, 
                                     n_samples=len(df_majority), 
                                     random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    return df_balanced

# this is just a function to print the ** over and under the EDA and so on to calculate and print it automatically.
def print_section(title, border_char='*'):
    border_length = len(title) + 4  # 2 spaces on each side
    border = border_char * border_length
    
    print(f"\n{border}")
    print(f"{border_char} {title} {border_char}")
    print(f"{border}")

# here we call out the main functions
def main():
    df = get_dataframe()
    
    # EDA
    print_section("EXPLORATORY DATA ANALYSIS (EDA)")
    print(print_dataset_info(df, 'Original'))
    
    df = get_dataframe()
    df_shuffled = preprocess_dataframe(df)
    df_val, df_tt = split_dataframe(df_shuffled)
    
    print(print_dataset_info(df_tt, 'Training/Testing'))

    plot_data(df_tt) # Reduced Dataset/ Training/Test Dataset
    
    # Data Transformation
    print_section("DATA TRANSFORMATION")
    print("\n->Multicolinearity Tests: Bartlett and KMO\n")
    multicolinearity_tests(df_tt)
    
    print("->Data Balancing:\n")
    check_class_distribution = class_distribution(df_tt, "original")
    df_balanced = balance_data(df_tt)
    balanced_class_distribution = class_distribution(df_balanced, "balanced")

    X = df_balanced[COLUMNS_OF_INTEREST]
    y = df_balanced[TARGET]
    
    print("->Principle Components Analysis (PCA):\n")   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_pca, X_test_pca, pca = pca_test(X_train, X_test)
    
    # Data Modelling
    print_section("DATA MODELLING (MODEL BUILDING)")
    accuracy, classification = train_and_evaluate_model(X_train_pca, X_test_pca, y_train, y_test)
    
if __name__ == "__main__":
    main()