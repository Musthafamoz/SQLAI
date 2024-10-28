import torch
from transformers import AutoModelForCausalLM, AutoTokenizerss
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
