import psycopg2
import pandas as pd
from sqlalchemy import create_engine

try:
    # Connect to the PostgreSQL database
    connection = psycopg2.connect(
        dbname="motoshop",
        user="postgres",
        password="admin",
        host="localhost",
        port="5432"
    )
    print("Connected to the database")

    engine = create_engine('postgresql://postgres:admin@localhost:5432/motoshop')

    # Read data from the tables using pandas
    df_customers = pd.read_sql('SELECT * FROM Customer LIMIT 5', connection)
    print("First 5 rows from Customer table:")
    print(df_customers.to_markdown())

    df_cars = pd.read_sql('SELECT * FROM cars LIMIT 5', connection)
    print("First 5 rows from cars table:")
    print(df_cars.to_markdown())

    df_car_sales = pd.read_sql('SELECT * FROM car_sales LIMIT 5', connection)
    print("First 5 rows from car_sales table:")
    print(df_car_sales.to_markdown())

    # What is the most expensive car?
    print('Most expensive car:')
    print(pd.read_sql('SELECT * FROM cars ORDER BY Price DESC LIMIT 1', engine))

    # What city has the most sales revenue?
    print('\nCity with most sales:')
    query = '''
    SELECT City, SUM(Total) AS Revenue
    FROM car_sales
    INNER JOIN Customer ON car_sales.CustomerID = Customer.CustomerID
    GROUP BY City
    ORDER BY Revenue DESC
    LIMIT 1
    '''
    print(pd.read_sql(query, engine))

    # Who is the best sales agent?
    print('\nBest sales agent:')
    query = '''
    SELECT SalesAgent, SUM(Total) AS Revenue
    FROM car_sales
    GROUP BY SalesAgent
    ORDER BY Revenue DESC
    LIMIT 1
    '''
    print(pd.read_sql(query, engine))

    # What is the most popular car?
    print('\nMost popular car:')
    query = '''
    SELECT Brand, Model, SUM(Quantity) AS Quantity
    FROM car_sales
    INNER JOIN cars ON car_sales.ProductID = cars.ProductID   
    GROUP BY Brand, Model
    ORDER BY Quantity DESC
    LIMIT 1
    '''
    print(pd.read_sql(query, engine))

    # Close the connection
    connection.close()
    print("Connection closed")

except psycopg2.Error as e:
    print("Error connecting to PostgreSQL database:", e)
