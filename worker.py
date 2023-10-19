import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

conversation_retrieval_chain = None
chat_history = []
llm = None
llm_embeddings = None

# Initialize the language model and its embeddings
def init_llm():
    global llm, llm_embeddings

    api_key=os.environ.get('OPENAI_KEY')
   
    # Initialize the embeddings for the language model
    llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo")
    llm_embeddings = OpenAIEmbeddings(openai_api_key = api_key)

# Process a PDF document
def process_document(document_path):
    global conversation_retrieval_chain, llm, llm_embeddings

    loader = PyPDFLoader(document_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Create a vector store from the document chunks
    db = Chroma.from_documents(texts, llm_embeddings)

    # Create a retriever interface from the vector store
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # Create a conversational retrieval chain from the language model and the retriever
    conversation_retrieval_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

# Process a user prompt
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history

    # Pass the prompt and the chat history to the conversation_retrieval_chain object
    result = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})

    chat_history.append((prompt, result['answer']))

    return result['answer']

# Initialize the language model
init_llm()
