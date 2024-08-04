# How to use this, where marked: 
# 1-2: PUT THE NAME OF THE DATABASE
# 2-2: PUT THE NAME OF THE TABLE
# so that you can retrieve the table.

import pandas as pd # to use a dataframe
from sqlalchemy import create_engine # to connect to SQL

# Database connection settings
db_config = {
    'host': '127.0.0.1',
    'user': 'root',         # Your MySQL username
    'password': '',         # Your MySQL password (empty by default in XAMPP)
    'database': 'database_4'  # 1-2: PUT THE NAME OF THE DATABASE HERE!!!
}

def get_dataframe():
    """
    Connect to the MySQL database, retrieves data, and return it as a Pandas DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the query results. Returns None if there is an error.
    
    Code Explanation:
        we wrap the procedure in a try-exception block. It tries to execute if successful,
        returns the dataframe. if not, returns the Exception Error.
    """
    try:
        # Create a SQLAlchemy engine: Connects to your database using connection details.
        engine = create_engine(f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")

        # 2-2: PUT THE NAME OF THE TABLE HERE!!!
        table_name = 'banknote_authentication'

        # Define your SQL query. 
        # # This is a command string to retrieve all records from a specified table.
        query = f"SELECT * FROM {table_name};"

        # Load data into a Pandas DataFrame
        df = pd.read_sql(query, engine)

        # Adjust Pandas display options
        pd.set_option('display.max_columns', None)  # Show all columns with no restrictions
        pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping to multiple lines

        return df

    except Exception as e:
        print(f"Error: {e}")
        return None
    
# this is just a local function that returns if the data was retrieved successfully or not
# it just shows a quick summary of it.
def import_dataset():
    df = get_dataframe()
    if df is not None:
        print("\nData retrieved successfully.")
        print(f"\nDataset Summary Information:")
        print(df.info())
    else:
        print("\nFailed to retrieve data.")

def main():
    import_dataset()

# this if-statement allows us to run our module directly.
# if this module is executed as the main program, 
# the main function will be executed.
if __name__ == "__main__":
    main()
