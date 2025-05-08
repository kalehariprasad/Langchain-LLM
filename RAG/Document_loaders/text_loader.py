from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader =TextLoader('RAG\Document_loaders\.machine_learningtxt')
llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )

model = ChatHuggingFace(llm = llm)
parser= StrOutputParser()

template = PromptTemplate(
    template='generate a summary on the {text}',
    input_variables=['text']
    )
chain = template|model|parser
docs= loader.load()
#print(docs[0])
#print(docs[0].page_content)
#print(docs[0].metadata)
print(chain.invoke({'text':docs[0].page_content}))