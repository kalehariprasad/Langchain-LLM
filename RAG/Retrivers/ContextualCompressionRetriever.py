from langchain_community.vectorstores import FAISS
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

# Initialize embedding model 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Set up the compressor using an LLM

llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )
model = ChatHuggingFace(llm=llm)
compressor = LLMChainExtractor.from_llm(model)

# Recreate the document objects from the previous data
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

# Create a FAISS vector store from the documents
vectorstore = FAISS.from_documents(docs, embedding_model)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create the contextual compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)
# Query the retriever
query = "What is photosynthesis?"
compressed_results = compression_retriever.invoke(query)

print("--- Result compressed_results ---")
for i, doc in enumerate(compressed_results):
    print(f"{i+1}",doc.page_content)