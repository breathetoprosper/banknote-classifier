from ucimlrepo import fetch_ucirepo
import pandas as pd
  
# fetch dataset 
banknote_authentication = fetch_ucirepo(id=267) 
  
# data (as pandas dataframes) 
X = banknote_authentication.data.features 
y = banknote_authentication.data.targets 
  
'''  
# metadata 
print(banknote_authentication.metadata) 
  
# variable information 
print(banknote_authentication.variables) 
'''

# Ensure the indices of X and y match
# This step is good practice to avoid alignment issues.
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Combine features and targets into a single DataFrame
# Rename the target column if necessary
df = pd.concat([X, y], axis=1)

print(df.head())

# Rename the column 'curtosis' to 'kurtosis'
df = df.rename(columns={'curtosis': 'kurtosis'})

print("\n", df.head())