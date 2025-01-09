import PyPDF2
from langchain.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint

def process_pdf(file):
    """Przetwarza plik PDF i zwraca tekst."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def create_embeddings(text):
    """Tworzy osadzenia dla tekstu."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # OpenAIEmbeddings()
    return embeddings.embed_documents([text])

def answer_question(question, retriever, history):
    """Generuje odpowiedź na pytanie, korzystając z retrievera i historii."""
    qa_chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(openai_api_key="TWÓJ_KLUCZ_API",temperature=0), retriever) # opcjonalnie HuggingFaceEndpoint(repo_id="google/flan-t5-xxl", temperature=0.5)
    return qa_chain.run({"question": question, "chat_history": history})
