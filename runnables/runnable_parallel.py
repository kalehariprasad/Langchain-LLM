from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel
from dotenv import load_dotenv

load_dotenv()


llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='write a tweet on the twitter about {topic}',
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template='write a linkedin post on the {topic}',
    input_variables=['topic']
)

tweet_chain = RunnableSequence(prompt1,model,parser)

linkedin_chain = RunnableSequence(prompt2,model,parser)

final_chain = RunnableParallel({'tweet':tweet_chain,'linkedin post':linkedin_chain})

result = final_chain.invoke({'topic':'data science hiring'})

print(result)
final_chain.get_graph().print_ascii()