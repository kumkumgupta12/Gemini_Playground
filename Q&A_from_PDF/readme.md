# Steps I followed to create this small Project

## What is our Aim
1. Pdf works as our knowledge base and we get answer from LLM , LLM tries to find answer from the given context/knowldeg base only. 
2. If knowledge base does not contain answer for given question , we do not want LLM to hallucinate .

## What are steps followed (technical)
1. Creating the Knowledge base
- when the pdfs are uploaded , we read it in chunks and create vector embeddings of it and store in **FAISS INDEX** .

2. Using langchain to create Q & A pipeline 
- where gemini can interact with multiple pdf files to generate the answer. (type = stuff)
- we have given a prompt template as well where context related instructions are already given to LLM . This instruction tries to avoid **hallucination**

3. Similarity search is performed
- When used has uploaded a question, we do similarity search from the created FAISS_index (we can store it in DB as well)
- we generate response by giving user question and knowledge base to the chunk(from step 2)

## User Interface
We have create a simple streamlit app for this purpose. 

## How to use this application
1. Clone this repository
2. create your virtual environment and install all dependencies
3. run command -  "streamlit run app.py" 
