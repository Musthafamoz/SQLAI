import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sqlparse
import psycopg2
import streamlit as st
import pandas as pd

# Function to execute SQL queries
def execute_query(query):
    conn = psycopg2.connect(
        dbname="kompany",
        user="postgres",
        password="admin",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    return cur, rows  # Return both cursor and rows

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
    CREATE TABLE IF NOT EXISTS Department (
    dept_id INT PRIMARY KEY,
    name VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS Employee (
    emp_id INT PRIMARY KEY,
    name VARCHAR(100),
    contact_number VARCHAR(20),
    email VARCHAR(100),
    dob DATE,
    address VARCHAR(255),
    dept_id INT,
    FOREIGN KEY (dept_id) REFERENCES Department(dept_id)
);

CREATE TABLE IF NOT EXISTS Project (
    project_id INT PRIMARY KEY,
    name VARCHAR(100),
    dept_id INT,
    project_manager_id INT,
    FOREIGN KEY (dept_id) REFERENCES Department(dept_id),
    FOREIGN KEY (project_manager_id) REFERENCES Employee(emp_id)
);

CREATE TABLE IF NOT EXISTS Task (
    task_id INT PRIMARY KEY,
    name VARCHAR(100),
    project_id INT,
    FOREIGN KEY (project_id) REFERENCES Project(project_id)
);

CREATE TABLE IF NOT EXISTS Employee_Task (
    emp_id INT,
    task_id INT,
    PRIMARY KEY (emp_id, task_id),
    FOREIGN KEY (emp_id) REFERENCES Employee(emp_id),
    FOREIGN KEY (task_id) REFERENCES Task(task_id)
);
s
CREATE TABLE IF NOT EXISTS Project_Manager (
    project_manager_id INT PRIMARY KEY,
    project_id INT,
    FOREIGN KEY (project_id) REFERENCES Project(project_id)
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
            cur, rows = execute_query(sql_query)  # Fetch cursor along with rows
            st.subheader("Query Results:")
            if len(rows) > 0:
                # Convert the result to a pandas DataFrame
                df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
                st.table(df)
            else:
                st.write("No results found.")

if __name__ == "__main__":
    main()
