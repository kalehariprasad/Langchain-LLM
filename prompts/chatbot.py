from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv(override=True)

llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )

model = ChatHuggingFace(llm=llm)


chat_history= []  # fro providing contex aware
while True:
    user_input = input('user_input :')
    chat_history.append(user_input) 

    if user_input == 'exit':
        break

    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print( 'AI :',result.content)

print(chat_history)