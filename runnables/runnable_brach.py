from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch,RunnablePassthrough,RunnableParallel ,RunnableSequence


load_dotenv()

llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

promt1 = PromptTemplate(
    template=' genarate a detailied report on {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template= 'summarise the {text}',
    input_variables=['text']
)

report_gen_chain = RunnableSequence(promt1,model,parser)

branch_chain= RunnableBranch(
    (lambda X :len(X.split())>300,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()

)

final_chain = RunnableSequence(report_gen_chain,branch_chain)

#print(report_gen_chain.invoke({'topic':'india in ICC tournaments'}))
print(final_chain.invoke({'topic':'india vs pakistan test macth'}))
final_chain.get_graph().print_ascii()