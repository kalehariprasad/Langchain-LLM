from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.messages import AIMessage, HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )

model = ChatHuggingFace(llm = llm)

chat_template = ChatPromptTemplate([
    ('system' , 'you are a helpfull {domain} expert'),
    ('human' , 'explain {topic} in simple and intutive manner ')
])

promt=chat_template.invoke({'domain':'cricket','topic':'No ball'})


print(promt)