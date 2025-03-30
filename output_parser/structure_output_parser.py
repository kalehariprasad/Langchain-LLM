from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()



llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )

model = ChatHuggingFace(llm=llm)



schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser =StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

#prompt = template.invoke({'topic':'cricket'})
#result = model.invoke(prompt)
#parser.parse(result)

chain = template|model|parser

result = chain.invoke({'topic':'cricket'})
print(result)

# advantage = can enforce  schema of output
# disadvantage data validtion is not possible in structure output parser