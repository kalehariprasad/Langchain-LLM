from langchain_community.retrievers import WikipediaRetriever
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace

retriever = WikipediaRetriever(top_k_results=2, lang="en")
# Define your query
query = "AHU systems in HVAC and its operation "

# Get relevant Wikipedia documents
docs = retriever.invoke(query)

# Print retrieved content
for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")  # truncate for display