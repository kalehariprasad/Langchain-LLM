from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community .document_loaders import PyPDFLoader


loader = PyPDFLoader('RAG\Document_loaders\Introduction_to_ML.pdf')

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=0,
)

# Perform the split
full_text = "".join([doc.page_content for doc in docs])
chunks = splitter.split_text(full_text)

print(len(chunks))
print(chunks[0])