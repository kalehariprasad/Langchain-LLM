from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv



load_dotenv()

loader = PyPDFLoader('RAG\Document_loaders\Introduction_to_ML.pdf')

docs = loader.load()

print(docs[0])