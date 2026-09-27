from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()  # Load environment variables from .env file
from langchain_mistralai import ChatMistralAI
model=ChatMistralAI(model="open-mistral-7b") 
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a professional Movie Information Extraction Assistant.

Your task:
Extract useful structured information from a movie paragraph and present it in a clean format.

Rules:
- Do NOT add explanations
- Do NOT add extra commentary
- Follow the exact format
- If information is missing → write NULL
- Keep summary short (2-3 lines max)
- Do NOT guess unknown facts

Output Format:
Movie Title:
Release Year:
Genre:
Director:
Main Cast:
Setting/Location:
Plot:
Themes:
Ratings:
Notable Features:

Short Summary:
"""),
    ("human",
     """Extract information from this paragraph:

{paragraph}             
""")#placeholder where movie will be added
])


paragraph = input("Enter a movie paragraph: ")

res= prompt.invoke({"paragraph": paragraph})
response=model.invoke(res)
print(response.content)
