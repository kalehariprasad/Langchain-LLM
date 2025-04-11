from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableLambda,RunnableParallel,RunnablePassthrough
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

def word_count(text):
    count =len(text.split())
    return count

chain = RunnableSequence(prompt1,model,parser)
parallel_chain =RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        #'explination':RunnableSequence(prompt2,model,parser),
        #'word_count': RunnableLambda (lambda X :len(X.split())),
        'word_count':RunnableLambda(word_count)
    }
)

final_chain = RunnableSequence(chain,parallel_chain)
result = final_chain.invoke({'topic':'harrypotter'})

print(f"the joke is {result['joke']} and lenth of the joke is{result['word_count']}")

final_chain.get_graph().print_ascii()