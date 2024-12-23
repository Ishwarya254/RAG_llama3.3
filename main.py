import os
import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


loader = PyPDFLoader(r'Your file path')
pages = loader.load_and_split()


# Access API keys from the .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Initialize Pinecone
pc = Pinecone(api_key='your PINECONE api key')
index_name = "pdf-question-answering"

# Ensure the index exists
try:
    index = pc.Index(index_name)
except Exception:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection="disabled"
    )
    index = pc.Index(index_name)


llm = ChatGroq(model_name = 'llama-3.3-70b-versatile')

pages[0]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

docs = text_splitter.split_documents(pages)
len(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


vector_store = PineconeVectorStore(embedding=embeddings, index=index)

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = vector_store.as_retriever())

query = "Type your question here!"

langchain.debug = True

chain({"question": query}, return_only_outputs=True)