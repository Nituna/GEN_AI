from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
model=init_chat_model("open-mistral-7b", model_provider="mistralai",temperature=0.9)
messages=[
    SystemMessage(content="You are a helpful assistant."),  #Means you should always be helpful to the user and give the best answer possible
]
#OR you can use dictionary too this is short term memory
print("-------WELCOME type 0 to exit-------") 
while True:      #This is used to keep the chat going until the user decides to exit
    #Simple prompt
    prompt=input("You: ")
    #messages.append(prompt)
    #System+user prompt
    messages.append(HumanMessage(content=prompt))
    if prompt=="0":
        break
    
    response=model.invoke(messages)
    #messages.append(response.content)
    messages.append(AIMessage(content=response.content))  #Object formate of appending the response to the messages list
    print("Bot: " + response.content)

    #Why we need to keep the history means if i ask chatbot to tell me your name that i gave before it cannot give becuase it does not have the history of the conversation so we need to keep the history of the conversation so that it can give the answer based on the previous conversation.
print("Message history: ",messages)  #This will print the entire conversation history between the user and the chatbot