#from langchain.chat_models import init_chat_model
# model = init_chat_model(
#   model_provider="huggingface",
 #   temperature=0.7,
  #  max_tokens=1024)

  #or
import os

from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if token:
    os.environ["HF_TOKEN"] = token
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1")
model=ChatHuggingFace(llm=llm)
response=model.invoke("Who are you?")
print(response.content)