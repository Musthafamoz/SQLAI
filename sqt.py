import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

cache_dir = "./model"

tokenizer = AutoTokenizer.from_pretrained("chatdb/natural-sql-7b")

model = AutoModelForCausalLM.from_pretrained(
    "chatdb/natural-sql-7b",
    device_map="auto",
    torch_dtype=torch.float16,
    cache_dir=cache_dir
)

questions = ['Show me the Total number of unique customers', 'Show me the Most expensive car', 'Find the total number of cars sold', 'Calculate the total discount amount given across all sales', 'Find the customers who have not made any purchases yet']

for question in questions:
    prompt = f"""
    ### Task 

    Generate a SQL query to answer the following question: `{question}` 
    
    ### PostgreSQL Database Schema 
    The query will run on a database with the following schema: 
    ```
    -- Customer table
CREATE TABLE Customer (
    CustomerID SERIAL PRIMARY KEY,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Email VARCHAR(100),
    PhoneNumber VARCHAR(20),
    Address VARCHAR(100),
    City VARCHAR(100),
    PostalCode VARCHAR(10),
    Country VARCHAR(50)
);

-- Cars table
CREATE TABLE cars (
    ProductID SERIAL PRIMARY KEY,
    Brand VARCHAR(50),
    Model VARCHAR(50),
    Year INTEGER,
    Price DECIMAL(10, 2)
);

-- Car sales table
CREATE TABLE car_sales (
    SalesID SERIAL PRIMARY KEY,
    CustomerID INTEGER,
    ProductID INTEGER,
    Quantity INTEGER,
    Price DECIMAL(10, 2),
    DiscountPercent INTEGER,
    Total DECIMAL(10, 2),
    SalesAgent VARCHAR(50),
    Date DATE
);

    ```
    
    ### Answer 
    Here is the SQL query that answers the question: `{question}` 
    ```sql
    """


    print("Question: " + question)
    print("SQL: ")

    # inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    generated_ids = model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=100001,
        pad_token_id=100001,
        max_new_tokens=400,
        do_sample=False,
        num_beams=1,
    
    )
    
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(outputs[0].split("```sql")[-1])
