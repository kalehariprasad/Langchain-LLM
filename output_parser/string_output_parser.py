from pydantic import BaseModel , Field
from typing import Optional,Literal
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from  langchain_core.prompts import PromptTemplate

load_dotenv()

llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )

model = ChatHuggingFace(llm=llm)


# 1st promt  detailed report
template1 = PromptTemplate(
    template = 'write a detailed report on {topic}',
    input_variables=['topic']

)

#2nd prompt genarate a summary

template2 = PromptTemplate(
    template = 'genarate a  6 line summmary on tyhe following text./{text}',
    input_variables= ['tesxt']
)

prompt1 =template1.invoke({'topic':'Data science'})
result =  model.invoke(prompt1)
prompt2 =template2.invoke({'text':result.content})
result2 = model.invoke(prompt2)
print(result2.content)