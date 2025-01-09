import PyPDF2
from langchain.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

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
    qa_chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), retriever)
    return qa_chain.run({"question": question, "chat_history": history})
