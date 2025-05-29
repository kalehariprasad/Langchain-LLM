from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = [
    Document(page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history.", metadata={"team": "Royal Challengers Bangalore"}),
    Document(page_content="Rohit Sharma is the most successful captain in IPL history.", metadata={"team": "Mumbai Indians"}),
    Document(page_content="MS Dhoni has led Chennai Super Kings to multiple IPL titles.", metadata={"team": "Chennai Super Kings"}),
    Document(page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket.", metadata={"team": "Mumbai Indians"}),
    Document(page_content="Ravindra Jadeja is a dynamic all-rounder for Chennai Super Kings.", metadata={"team": "Chennai Super Kings"}),
]

#  Build vector store
vector_store = FAISS.from_documents(docs, embedding=embeddings)

#  Similarity search using MMR (for diverse results)
similarity_query = vector_store.max_marginal_relevance_search(
    query='Who among these are a batter?',
    k=1
)

# Metadata filtering: all Mumbai Indians players
MI_team = vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Mumbai Indians"}
)

print("MMR Query:", similarity_query)
#print("Mumbai Indians Filtered Query:", MI_team)
