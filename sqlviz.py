import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sqlparse
import psycopg2
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to execute SQL queries
def execute_query(query):
    conn = psycopg2.connect(
        dbname="motoshop",
        user="postgres",
        password="admin",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    column_names = [desc[0] for desc in cur.description]  # Fetch column names from cursor description
    conn.close()  # Close connection
    return column_names, rows  # Return both column names and rows

# Function to generate SQL query from natural languages
def generate_sql(question):
    cache_dir = "./model"
    tokenizer = AutoTokenizer.from_pretrained("chatdb/natural-sql-7b")
    model = AutoModelForCausalLM.from_pretrained("chatdb/natural-sql-7b", cache_dir=cache_dir)

    # Prompt setup
    prompt = f"""
    ### Task 

    Generate a SQL query to answer the following question: `{question}` 
    
    ### PostgreSQL Database Schema 
    The query will run on a database with the following schema: 
    ```
    -- Customer table
CREATE TABLE customer (
    customerid SERIAL PRIMARY KEY,
    firstname VARCHAR(50),
    lastname VARCHAR(50),
    email VARCHAR(100),
    phonenumber VARCHAR(20),
    address VARCHAR(100),
    city VARCHAR(100),
    postalcode VARCHAR(10),
    country VARCHAR(50)
);

-- Cars table
CREATE TABLE cars (
    productid SERIAL PRIMARY KEY,
    brand VARCHAR(50),
    model VARCHAR(50),
    year INTEGER,
    price DECIMAL(10, 2)
);

-- Car sales table
CREATE TABLE car_sales (
    salesid SERIAL PRIMARY KEY,
    customerid INTEGER,
    productid INTEGER,
    quantity INTEGER,
    price DECIMAL(10, 2),
    discountpercent INTEGER,
    total DECIMAL(10, 2),
    salesagent VARCHAR(50),
    date DATE
);

    ```
    
    ### Answer 
    Here is the SQL query that answers the question: `{question}` 
    ```sql
    """.format(question=question)
    messages = [{'role': 'user', 'content': prompt}]

    # Model generation parameters
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            do_sample=False,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )

    # Extract SQL query from model output
    sql_query = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return sql_query

# Streamlit UI
def main():
    st.title("Natural Language to SQL Query Converter")
    question = st.text_input("Enter your question:")
    if st.button("Generate SQL"):
        if question:
            sql_query = generate_sql(question)
            st.subheader("Generated SQL Query:")
            st.code(sqlparse.format(sql_query, reindent=True), language='sql')
            column_names, rows = execute_query(sql_query)  # Fetch column names along with rows
            st.subheader("Query Results:")
            if len(rows) > 0:
                # Convert the result to a pandas DataFrame
                df = pd.DataFrame(rows, columns=column_names)
                st.table(df)
                
                # Plotting results
                st.subheader("Visualization of Query Results:")
                plot_type = st.selectbox("Select plot type", ["Bar Plot", "Line Plot", "Pie Chart"])
                
                if plot_type == "Bar Plot":
                    st.bar_chart(df)
                elif plot_type == "Line Plot":
                    st.line_chart(df)
                elif plot_type == "Pie Chart":
                    pie_column = st.selectbox("Select column for Pie Chart", column_names)
                    st.write(df[pie_column].value_counts().plot.pie(autopct='%1.1f%%'))
                
            else:
                st.write("No results found.")

if __name__ == "__main__":
    main()
