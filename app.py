import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat",page_icon=":books:")

    st.header("Chat do poznawania zasad gier planszowych")
    st.text_input("Zapytaj o zasady gry:")

    with st.sidebar:
        st.subheader("Twoje dokumenty")
        pdf_docs = st.file_uploader(
            "Wczytaj swoje pliki PDF i kliknij przycisk 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # wczytywane pdf
                raw_text = get_pdf_text(pdf_docs)

                #kawałki tekstu
                text_chunks = get_text_chunks(raw_text)

                #vectorstore
                vectorstore = get_vectorstore(text_chunks)




if __name__ == '__main__':
    main()
