from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
#HuggingFacePipeline is a pipline which helps to run the model locally without using the huggingface api and can download by hugging face website 
llm =HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",task="text-generation",
    pipeline_kwargs=dict(max_new_tokens=100,do_sample=False,repetition_penalty=1.2
    ))
chat_model=ChatHuggingFace(llm=llm)
result=chat_model.invoke("what is your thinking about genai?")
print(result.content)