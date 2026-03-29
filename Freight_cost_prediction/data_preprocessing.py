#I created modular functions to load data from a SQLite database, prepare features by separating input and target variables, and split the dataset into training and testing sets to evaluate model performance effectively

import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split

def load_vendor_invoice_data(db_path: str): #db_path-path of the db and :str it  should be in string format
    conn=sqlite3.connect(db_path)
    query="select * from vendor_invoice"
    df=pd.read_sql_query(query,conn)
    conn.close() #close the connection as it frees the memory
    return df #return the df .. giev em the df so i can use it later

def prepare_features(df: pd.DataFrame):
    """select features and target variables"""
    X=df[['Dollars']] #Double brackets → DataFrame (2D)
    y=df['Freight']
    return X,y
#we split the data just to check if our model perfroms well on unseen data or not
def split_data(X,y,test_size=0.2,random_state=42):
    """split dataset into train and test data"""
    return train_test_split(X,y,test_size=test_size,random_state=random_state)