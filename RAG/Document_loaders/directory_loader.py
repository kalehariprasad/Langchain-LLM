from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv



load_dotenv()

loader = DirectoryLoader(
    path="RAG\Document_loaders\Books",
    glob="*.pdf",
    loader_cls=PyPDFLoader)

docs = loader.load()
print(len(docs))
print(docs[470].metadata)

for documents in docs:
   print(documents.metadata)