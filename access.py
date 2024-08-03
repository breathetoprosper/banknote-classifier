# How to use this, where marked: 
# 1-2: PUT THE NAME OF THE DATABASE
# 2-2: PUT THE NAME OF THE TABLE
# so that you can retrieve the table.

import pandas as pd
from sqlalchemy import create_engine

# Database connection settings
db_config = {
    'host': '127.0.0.1',
    'user': 'root',         # Your MySQL username
    'password': '',         # Your MySQL password (empty by default in XAMPP)
    'database': 'database_4'  # 1-2: PUT THE NAME OF THE DATABASE HERE!!!
}

def get_dataframe():
    """
    Connect to the MySQL database, retrieve data, and return it as a Pandas DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the query results. Returns None if there is an error.
    """
    try:
        # Create a SQLAlchemy engine: Connects to your database using connection details.
        engine = create_engine(f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")

        # 2-2: PUT THE NAME OF THE TABLE HERE!!!
        table_name = 'banknote_authentication'

        # Define your SQL query. Command string to retrieve all records from a specified table.
        query = f"SELECT * FROM {table_name};"

        # Load data into a Pandas DataFrame
        df = pd.read_sql(query, engine)

        # Adjust Pandas display options
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping to multiple lines

        return df

    except Exception as e:
        print(f"Error: {e}")
        return None
    
def import_dataset():
    df = get_dataframe()
    if df is not None:
        print("\nData retrieved successfully.")
    else:
        print("\nFailed to retrieve data.")

if __name__ == "__main__":
    # If this script is run directly, display the DataFrame
    df = get_dataframe()
    if df is not None:
        print("\nData retrieved successfully.")
        print(df)
        print(df.info())     
        print(df.describe(include='all'))
    else:
        print("\nFailed to retrieve data.")
