from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )

model = ChatHuggingFace(llm=llm)

chat_template = ChatPromptTemplate([
    ('system','you are a helpfull customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', 'where is my refund')

    ])

# load Chat_history
chat_history = []
with open('prompts\chat_history.txt') as f:
    chat_history.extend(f.readlines())
#print(chat_history)

promt = chat_template.invoke({'chat_history':chat_history,'query':'where is my refund'})
