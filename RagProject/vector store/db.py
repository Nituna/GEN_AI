#from langchain_community.vectorstores import Chroma

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
vectorstore=Chroma.from_documents(
    documents=docs, 
    embedding=embeddings,
    persist_directory="chroma_db") #by this chroma_db folder will be created in the current directory and the data will be stored in it
result=vectorstore.similarity_search("what is used for data analysis",k=2)#this will return the top 2 documents that are most similar to the query "what is used for data analysis"
#This vectore store is working as retriever why retiever needed it is beacuse there is a speciality in retrever they have multiple searching technique

for r in result:
    print(r.page_content)

retriever=vectorstore.as_retriever()
docs=retriever.invoke("explain dl")  #i can invoke retriever because it is runnable but not vectorstore

for d in docs:
    print(d.page_content)