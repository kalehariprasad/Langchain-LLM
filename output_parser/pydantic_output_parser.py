from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()

llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )

model = ChatHuggingFace(llm=llm)


class person (BaseModel):
    name :str =Field(description='name of the person')
    age :int =Field(gt=18,description= 'age of the person')
    city : str = Field(description= ' name of the city person belongs to')

parser =PydanticOutputParser(pydantic_object= person)


# Define the PromptTemplate
template = PromptTemplate(
    template="Generate name, age, city of the fictional {place}. \n{formate_instructions}",
    input_variables=['place'],
    partial_variables={'formate_instructions': parser.get_format_instructions()},
)
 
#prompt = template.invoke({'place':'india})
#print(prompt)
#result = model.invoke(prompt)
#print(result.content)
chain = template|model|parser

result = chain.invoke({'place':'indian'})

print(result)

# this kind of prompt will not only provide structured output but also validate formate of data types