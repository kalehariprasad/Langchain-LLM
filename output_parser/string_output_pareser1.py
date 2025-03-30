from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from  langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    template = 'genarate a  5 line summmary on tyhe following text./{text}',
    input_variables= ['tesxt']
)


parser= StrOutputParser()

chain = template1|model|parser|template2|model|parser

result = chain.invoke({'topic':'AI'})
print(result)