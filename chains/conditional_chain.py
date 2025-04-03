from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain.schema.runnable import RunnableBranch,RunnableLambda
from pydantic import BaseModel ,Field
from typing import Literal

#RunnableParallel parameter is used for runninfg parrele chains at a time 
load_dotenv()


llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )
model = ChatHuggingFace(llm=llm)


class feedback(BaseModel):
    sentiment:Literal['positive','negitive']=Field(description='give the sentiment of the feedback')

parser = StrOutputParser()
pydatic_parser = PydanticOutputParser(pydantic_object=feedback)

promt1 =PromptTemplate (
    template='clasiify the sentiment of the following feedback into positive or negitive\{feedback} \n {formate_instruction}',
    input_variables=['feedback'],
    partial_variables= {'formate_instruction':pydatic_parser.get_format_instructions()}
)

promt2 = PromptTemplate(
    template='write an appropiate responce to the positive {feedback}',
    input_variables=['feedback']
)

promt3 = PromptTemplate(
    template= 'write an appropiate responce to the negitive {feedback}',
    input_variables=['feedback']
)
classifier_cahin =promt1|model|pydatic_parser

classifier_cahin.invoke({'feedback':'this is awesome shoes'})


branch_chain = RunnableBranch(
   (lambda X: X.sentiment == 'positive', promt2 | model | parser),  # X will be 'positive' or 'negative'
   (lambda X: X.sentiment == 'negitive', promt3 | model | parser),  # X will be 'positive' or 'negative'
   RunnableLambda(lambda X: 'Could not find sentiment')
)

final_chain = classifier_cahin|branch_chain
#result = classifier_cahin.invoke({'feedback':'this is awesome shoes'})
result = final_chain.invoke({'feedback':'this is terrable shoes'})
print(result)