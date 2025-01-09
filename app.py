import streamlit as st
from utils import process_pdf, create_embeddings, answer_question
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings 

# Ustawienia Streamlit
st.set_page_config(page_title="Chatbot PDF RAG", layout="wide")
st.title("Chatbot PDF z RAG")

# Przesyłanie pliku PDF
uploaded_file = st.file_uploader("Prześlij plik PDF", type="pdf")

if uploaded_file:
    with st.spinner("Przetwarzanie PDF..."):
        text = process_pdf(uploaded_file)
        st.success("Plik PDF przetworzony!")
    
    # Generowanie osadzeń
    if st.button("Stwórz osadzenia"):
        with st.spinner("Tworzenie osadzeń..."):
            embeddings = create_embeddings(text)
            vector_store = FAISS.from_texts([text], HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")) # inny embedding OpenAIEmbeddings(openai_api_key="TWÓJ_KLUCZ_API")
            st.success("Osadzenia stworzone!")
    
    # Historia rozmowy
    history = st.session_state.get("history", [])
    
    # Zadawanie pytań
    question = st.text_input("Zadaj pytanie dotyczące pliku PDF")
    if question:
        with st.spinner("Generowanie odpowiedzi..."):
            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
            retriever = vector_store.as_retriever()
            qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)
            answer = qa_chain.run({"question": question, "chat_history": history})
            history.append((question, answer))
            st.write(answer)

    # Wyświetlanie historii
    if st.checkbox("Pokaż historię rozmowy"):
        for q, a in history:
            st.write(f"**Pytanie:** {q}")
            st.write(f"**Odpowiedź:** {a}")
