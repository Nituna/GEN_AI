'''
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader #impleamneting the text loader into the main function
from langchain_community.document_loaders import PyPDFLoader #impleamneting the pdf loader into the main function
from langchain_text_splitters import RecursiveCharacterTextSplitter #impleamneting the text splitter into the main function
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
#data=TextLoader("document loaders/notes.txt",encoding="utf8")
#docs=data.load()
#data1=PyPDFLoader("document loaders/GRU.pdf")


data=PyPDFLoader("document loaders/deeplearning.pdf")
docs=data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200)
chunks=splitter.split_documents(docs)

#Prompt Template
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarizes the pdf provided by the user"),
    ("user", "{data}")
])

model = ChatMistralAI(model="open-mistral-7b")

#prompt=template.format_messages(data=docs[0].page_content)
#res=model.invoke(prompt)
#print(res.content)

prompt=template.format_messages(data=docs[0].page_content) #we can send only one page at a time or else we will get context window error(context length exceeded error)
res=model.invoke(prompt)
print(res.content)
'''
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore=Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)
retreiver=vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={
        "k" : 4,
        "fetch_k" : 10, #It means first you will apply similarity search then you get 10 after apply mmr you get 4
        "lambda_mult" : 0.5    #If lambda is 1 less diversity(means retrieve the answer related to query) for 0 vise versa
    }
)

llm=ChatMistralAI(model="open-mistral-7b")

#Prompt template
prompt=ChatPromptTemplate.from_messages(
    [
           (
            "system",
            """You are a helpful AI assistant.
            Use ONLY the provided context to answer the question.
            If the answer is not present in the context,
            say: "I could not find the answer in the document."
            """
            ),
           (
            "human",
            """Context:
            {context}
            Question:
            {question}
            """
        ) 
    ]
)
print("Rag sys created")
print("press 0 to exit")
while True:
    query=input("YOU: ")
    if query=='0':
        break
    docs=retreiver.invoke(query)
    context="".join(
        [doc.page_content for doc in docs]
    )

    final_prompt=prompt.invoke({
        "context":context,
        "question":query
    })

    response=llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")