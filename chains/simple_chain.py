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


prompt = PromptTemplate(
    template= 'genarate 5 facts about{topic}',
    input_variables=['topic']
)

parser =StrOutputParser()


chain = prompt |model |parser

result = chain.invoke({'topic':'cricket'})
print(result)
chain.get_graph().print_ascii()   #by usnig this code you can visualise your chain 