# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import google.generativeai as genai
import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from mixedbread_ai.client import MixedbreadAI
import os

google_api_key = os.getenv('GOOGLE_API_KEY')



mxbai = MixedbreadAI(api_key="emb_3835f9d13d307758dcb8ae0738ea3bce4bedc1601d3543a9")

embeddings = mxbai.embeddings(
  model='mixedbread-ai/mxbai-embed-large-v1')


# embeddings = HuggingFaceEmbeddings(
#         model_name="mixedbread-ai/mxbai-embed-large-v1",
#         model_kwargs={'device': 'cpu'})

# llm = ChatOllama(model = 'llama3:8b')

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.0,
    max_tokens=None,
    timeout=None, 
    google_api_key = 'AIzaSyCQbNkygleMD3b6QI1QFq8-Zr9gpMBAfP4'  
)
# vectordb = Chroma(persist_directory="krushimitra/qadb111", embedding_function=embeddings)
vectordb = Qdrant.from_existing_collection(
    embedding= embeddings,
    collection_name="KrushiMitra",
    url="https://75b181e0-aa32-4dcd-8ef1-2384c1096d63.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="1E8QJAmx7z8mjS8ygqRK5FsEVP-4bPA8hAGujsIFnqDX1vuEiuQNOg")

retriever = vectordb.as_retriever(search_type="mmr",search_kwargs={'k': 10, 'lambda_mult': 0.5})

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """You are KrushiMitra an expert assistant specializing in agriculture and farming. 
Use the following pieces of retrieved context to provide accurate, practical, and helpful answers to the questions. 
Your responses should be helpful , clear, and relevant to farmers' needs.Only use English.
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
