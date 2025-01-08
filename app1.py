import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# 1) Use the *official* LangChain integrations for Google PaLM:
from langchain.embeddings import GooglePalmEmbeddings
from langchain.chat_models import ChatGooglePalm

from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # Adjust chunk_overlap if needed (1000 overlap is quite large)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # 2) Use GooglePalmEmbeddings instead of GoogleGenerativeAIEmbeddings
    embeddings = GooglePalmEmbeddings(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model_name="models/embedding-gecko-001"  # or "models/embedding-bison-001"
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    # 3) Create a custom prompt template
    prompt_template = """
    Answer the question as accurately and with as much detail as possible 
    from the provided context. If the answer is not present in the provided 
    context, respond with: "answer is not available in the context."
    
    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # 4) Use ChatGooglePalm instead of ChatGoogleGenerativeAI
    model = ChatGooglePalm(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model_name="models/chat-bison-001",  # Public model name
        temperature=0.3
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    # Re-load the FAISS index using the same embeddings
    embeddings = GooglePalmEmbeddings(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model_name="models/embedding-gecko-001"
    )

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("chat with multiple PDFs")
    st.header("Chat with PDF using ChatGooglePalm")

    user_question = st.text_input("Ask a question from the PDF(s)")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF(s)", accept_multiple_files=True)
        if st.button("Submit & Proceed"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done!")

if __name__ == "__main__":
    main()
