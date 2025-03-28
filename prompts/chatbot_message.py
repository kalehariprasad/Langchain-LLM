from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )

model = ChatHuggingFace(llm=llm)

messgaes = [
    SystemMessage(content = 'you are a helpfull assistance '),
    ]
while True:
    user_input =input("user_input :")
    messgaes.append(HumanMessage(content=user_input))
    if user_input =="exit":
        break
    result = model.invoke(messgaes)
    print('AI :',result.content)
    messgaes.append(AIMessage(content=result.content))


print(messgaes)