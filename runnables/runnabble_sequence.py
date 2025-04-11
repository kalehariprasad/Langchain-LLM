from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()


llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )

model = ChatHuggingFace(llm = llm)


prompt1 = PromptTemplate( 
    template='write a joke on the {topic}',
    input_variables = ['topic']
)

prompt2 =PromptTemplate(
    template= "explain the {joke}",
    input_variables=['joke']
    )

parser =StrOutputParser()

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

result = chain.invoke({'topic':'spiderman'})

print(result)