from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

url= "https://en.wikipedia.org/wiki/Narendra_Modi"
loader = WebBaseLoader(url)
docs = loader.load()

promt = PromptTemplate(
    template= 'answer the following {question} ',
    input_variables=['question']
)

llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = promt|model|parser

print(chain .invoke({'question':"what is the date of birth of modi"}))