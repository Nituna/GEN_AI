
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initializeing the Gemini embedding class
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2",dimensions=64,google_api_key="Your_Google_API_Key_Here")  # Replace with your actual Google API key

#Texts=["Plan me a trip to bali",
        #"Write a poem about AI in 4 lines",
        #"What is your thinking about genai?"]
#vector = embeddings.embed_documents(Texts)
vector = embeddings.embed_query("Plan me a trip to bali")
print(vector)