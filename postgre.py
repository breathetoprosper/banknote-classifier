import pandas as pd
from sqlalchemy import create_engine

# Database connection settings for PostgreSQL
db_config = {
    'host': 'localhost',        # PostgreSQL server address
    'user': 'postgres',         # PostgreSQL username
    'password': 'ADMIN',        # PostgreSQL password
    'database': 'db2',          # 1-2: PUT THE NAME OF THE DATABASE HERE!!!
    'port': '5432'              # PostgreSQL server port
}

def get_dataframe():
    """
    Connect to the PostgreSQL database, retrieve data, and return it as a Pandas DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the query results. Returns None if there is an error.
    """
    try:
        # Create a SQLAlchemy engine for PostgreSQL
        engine = create_engine(f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")

        # 2-2: PUT THE NAME OF THE TABLE HERE!!!
        table_name = 'banknote_authentication'

        # Define your SQL query
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

if __name__ == "__main__":
    # If this script is run directly, display the DataFrame
    df = get_dataframe()
    if df is not None:
        print("\n", df)
        print("\n", df.info())
        print("\n", df.describe(include='all'))
    else:
        print("\nFailed to retrieve data.")