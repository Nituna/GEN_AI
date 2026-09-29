#load pdf
#split pdf into chunks
#create the embedding
#store in chroma db
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

data=PyPDFLoader("document loaders/deeplearning.pdf")
docs=data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200)  #function for splitting
chunks=splitter.split_documents(docs) #splitting happens here

#embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2") #its limit excedded so i created through hugging face
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)