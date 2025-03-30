from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from  langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()
template = PromptTemplate(
    template= 'give me name ,age ,ciy of a company in AI field./n {formate_instruction} ',
    input_variables=[],
    partial_variables= {'formate_instruction':parser.get_format_instructions()}
)
template2 = PromptTemplate(
    template= 'give me 5 facts about{topic}./n {formate_instruction} ',
    input_variables=['topic'],
    partial_variables= {'formate_instruction':parser.get_format_instructions()}
)  # inthis method of parsing we cant define thw structure formate  and responce schema

prompt = template.format()
#result =model.invoke(prompt)
#print(result.content)

chain = template2|model|parser

result = chain.invoke({"topic":"machine learning"})

print(result)