from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()
# Relevant health & wellness documents

all_docs = [
    # Relevant Health & Wellness Documents
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="A balanced diet with whole grains can stabilize blood sugar and improve gut health.", metadata={"source": "H6"}),
    Document(page_content="Stretching in the morning enhances circulation and reduces muscle tension.", metadata={"source": "H7"}),
    Document(page_content="Limiting screen time before bed supports melatonin production and sleep quality.", metadata={"source": "H8"}),
    Document(page_content="Strength training supports bone density and helps prevent age-related muscle loss.", metadata={"source": "H9"}),
    Document(page_content="Spending time in nature can reduce anxiety and elevate overall mood.", metadata={"source": "H10"}),
    Document(page_content="Daily meditation can reduce anxiety and increase gray matter density in the brain.", metadata={"source": "H11"}),
    Document(page_content="Probiotic-rich foods like yogurt support healthy digestion and immune function.", metadata={"source": "H12"}),
    Document(page_content="Exposure to morning sunlight helps regulate circadian rhythms and boosts vitamin D.", metadata={"source": "H13"}),
    Document(page_content="Regular cardiovascular exercise strengthens the heart and reduces cholesterol levels.", metadata={"source": "H14"}),
    Document(page_content="Limiting processed sugar intake can reduce inflammation and improve skin clarity.", metadata={"source": "H15"}),

    # Non-Relevant (Distractor) Documents
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
    Document(page_content="Quantum computing uses qubits to process complex algorithms beyond classical systems.", metadata={"source": "I6"}),
    Document(page_content="Venus rotates in the opposite direction to most planets in our solar system.", metadata={"source": "I7"}),
    Document(page_content="The Eiffel Tower was originally constructed as a temporary structure for a world's fair.", metadata={"source": "I8"}),
    Document(page_content="Blockchain ensures data integrity through decentralized consensus mechanisms.", metadata={"source": "I9"}),
    Document(page_content="Modern drones utilize gyroscopes and GPS for stable autonomous flight.", metadata={"source": "I10"}),
    Document(page_content="Artificial intelligence is revolutionizing customer service through chatbots.", metadata={"source": "I11"}),
    Document(page_content="The Great Wall of China stretches over 13,000 miles and is visible from space.", metadata={"source": "I12"}),
    Document(page_content="Machine learning algorithms are used in fraud detection for financial institutions.", metadata={"source": "I13"}),
    Document(page_content="Shakespeare’s influence on the English language includes over 1,700 new words.", metadata={"source": "I14"}),
    Document(page_content="The periodic table organizes elements by atomic number and chemical properties.", metadata={"source": "I15"}),
]

# Initialize embedding model & model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )
model = ChatHuggingFace(llm=llm)

# Create FAISS vector store
vectorstore = FAISS.from_documents(documents=all_docs, embedding=embedding_model)
# Create retrievers
similarity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=model
)
# Query
query = "How to improve energy levels and maintain balance?"

# Retrieve results
similarity_results = similarity_retriever.invoke(query)
multiquery_results= multiquery_retriever.invoke(query)

#print(similarity_results)
#print(multiquery_results)

for i, doc in enumerate(similarity_results):
    print(f"\n--- similarity_results {i+1} ---")
    print(doc.page_content)

for i, doc in enumerate(multiquery_results):
    print(f"\n--- multiquery_results {i+1} ---")
    print(doc.page_content)
    # mistralai/Mistral-7B-Instruct-v0.3  this model is not performing well on multiquery_retriever .
    # its giving some irrelevent answers you csn try any open AI model