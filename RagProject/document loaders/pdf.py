#Pdf loader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

data=PyPDFLoader("document loaders/GRU.pdf")
docs=data.load()
#tiktoken a token based splitter used in openai chatgpt model

#splitter = TokenTextSplitter(
#    chunk_size=1000, 
#    chunk_overlap=10)
#chunks = splitter.split_documents(docs)
#print(len(docs))
#print(len(chunks))
#print(chunks[0].page_content)

#splitting recursively


