from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )
model = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template= 'genarate text about the {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template= 'cereate  a summaray of the {text }',
    input_variables=['text']
)


parser =StrOutputParser()


chain = prompt1 |model |parser |prompt2 |model|parser

result = chain.invoke({'topic':'movie'})
print(result)
chain.get_graph().print_ascii()   #by usnig this code you can visualise your chain 