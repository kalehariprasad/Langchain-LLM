from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough,RunnableSequence,RunnableParallel

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

joke_chain = RunnableSequence(prompt1,model,parser)

summary_cahin = RunnableParallel(
    {
        'joke' : RunnablePassthrough(),
        'exaplinatioon' : RunnableSequence(prompt2,model,parser)
    }
)

final_chain = RunnableSequence(joke_chain,summary_cahin)

result = final_chain.invoke({'topic':'movie'})
print(result)
final_chain.get_graph().print_ascii()