from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model  #this line is by langchain docs -> models -> initialize a model 
#model=init_chat_model("gemini-3.8-flash",model_provider="google_genai")  #In langchain you can use different models 
#model=init_chat_model("openai/gpt-oss-120b", model_provider="groq")   #openai/gpt-oss-120b", model_provider="groq or groq:openai/gpt-oss-120b
model=init_chat_model("open-mistral-7b", model_provider="mistralai",temperature=0.9,max_tokens=20)  #mistral-7b-instruct, model_provider="mistral
response=model.invoke("write a poem about AI in 4 lines")
print(response.content)