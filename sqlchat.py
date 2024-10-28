import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sqlparse
import psycopg2

cache_dir = "./model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("chatdb/natural-sql-7b")
model = AutoModelForCausalLM.from_pretrained("chatdb/natural-sql-7b", cache_dir=cache_dir)

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
    conn.close()
    return rows

questions = [
    # 'Show me the Total number of unique customers',
    'Show me the Most expensive car',
    # 'Find the total number of cars sold',
    # 'Calculate the total discount amount given across all sales',
    # 'Find the customers who have not made any purchases yet'
]

for question in questions:
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

    print("Question: " + question)
    print("SQL: ")

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
    print("Question: " + question)
    print("Generated SQL: " + sqlparse.format(sql_query, reindent=True))

    # Execute SQL query against the database
    rows = execute_query(sql_query)
    print("Result: ")
    for row in rows:
        print(row)

# Clear CPU memory
torch.cuda.empty_cache()
