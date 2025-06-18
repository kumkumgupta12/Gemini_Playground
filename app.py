# import streamlit as st
# from PyPDF2 import PdfReader

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os 

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.chains.question_answering import load_qa_chain
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains.question_answering import load_qa_chain helps to do chat
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import io


# load_dotenv()

# genai.configure(api_key = os.getenv("Google_API"))

# # get text from pdf
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         if isinstance(pdf, bytes):
#             pdf_stream = io.BytesIO(pdf)
#         else:
#             pdf_stream = pdf  # assume it's already a file-like object
#         pdf_reader= PdfReader(pdf_stream)

#         for page in pdf_reader.pages:
#             text += page.extract_text()

#     return text

# # get chunks from text
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# # do vector embeddings
# def get_vectors(chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "model/embedding-001")
#     vector_store = FAISS.from_texts(chunks,embedding = embeddings)
#     vector_store.save_local("faiss_index/")  # save in local

# def get_conversational_chain():
#     prompt_template= """
#     Answer the question in depth and detailed from provided context, make sure to provide answer in context
#     if Answer is not provided in context just tell answer is not available in the given context.
#     Don't provide the wrong answer. 
#     Context : \n {context} \n
#     Question : \n{question}? \n
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature = 0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables = ['context', 'question'])
#     # load_qa_chain to create Question answer pipeline with a LLM+prompt+chain_type
#     chain= load_qa_chain(model, chain_type = "stuff", prompt = prompt)
#     return chain


# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#     # loading the faiss index where all text embeddings are stored locally
#     new_db = FAISS.load_local("faiss_index", embeddings)

#     # do similarity search based on question
#     docs = new_db.similarity_search(user_question)
    
#     # trying to get the response/answer
#     chain = get_conversational_chain()

#     response = chain(
#         {"input_documents": docs, "question": user_question}
#         , return_only_outputs= True
#     )

#     print(response)
#     st.write("Reply:", response["output_text"])


# def main():
#     st.set_page_config("Get Answers from multiple PDF")
#     st.header("Get Answers from multiple PDF")

#     user_question = st.text_input("Ask a question from PDF file context")

#     if user_question:
#         user_input(user_question)
    
#     # converting pdf to vector embeddings (to creat FAISS index)
#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF, submit and get response")
#         if st.button("Submit"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vectors(text_chunks)
#                 st.success("Done")

# if __name__== "__main__":
#     main()


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import io

load_dotenv()
genai.configure(api_key=os.getenv("Google_API"))

api_key = os.getenv("Google_API")



# Extract text from multiple PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_stream = io.BytesIO(pdf.read())
        pdf_reader = PdfReader(pdf_stream)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# Generate embeddings and save FAISS index
def get_vectors(chunks):
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index/")

# Create QA chain with Gemini and custom prompt
def get_conversational_chain():
    prompt_template = """
    Answer the question in detail from the provided context.
    If the answer is not in the context, say: "Answer not found in the provided context."
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro",google_api_key=api_key, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Handle user questions
def user_input(user_question):
    try:
        # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
        new_db = FAISS.load_local("faiss_index", embeddings,  allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("🔎 **Answer:**", response["output_text"])
    except Exception as e:
        st.error(f"Error: {e}")

# Streamlit App
def main():
    st.set_page_config("Multi-PDF QnA with Gemini")
    st.header("📄 Ask Questions from Multiple PDFs")

    user_question = st.text_input("Ask a question based on uploaded PDFs:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📚 Upload PDFs")
        pdf_docs = st.file_uploader("Upload one or more PDF files", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit"):
            if pdf_docs:
                with st.spinner("📖 Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    get_vectors(chunks)
                    st.success("✅ FAISS Index Created!")
            else:
                st.warning("Please upload at least one PDF.")

if __name__ == "__main__":
    main()
