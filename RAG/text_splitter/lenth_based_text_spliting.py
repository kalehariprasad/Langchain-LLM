from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('RAG\Document_loaders\Introduction_to_ML.pdf')

docs = loader.load()

text = """
The Indian Premier League (IPL) is a professional Twenty20 cricket league in India, officially known as the TATA IPL for sponsorship reasons. Founded by the Board of Control for Cricket in India (BCCI) in 2007, the league has grown to become one of the most popular and financially successful cricket tournaments in the world. The IPL features franchise teams representing different Indian cities and is typically held between March and May each year.

Each IPL team is composed of domestic and international players, and team rosters are determined through auctions that generate significant media and public interest. The format includes a round-robin group stage followed by playoffs and a final to determine the champion. Over the years, the IPL has seen legendary performances from players like MS Dhoni, Virat Kohli, Rohit Sharma, and international stars such as AB de Villiers, Chris Gayle, and David Warner.

The IPL has played a significant role in revolutionizing the sport with innovations like cheerleaders, strategic time-outs, and fan engagement through social media. It also offers a lucrative platform for young Indian cricketers to showcase their talent and secure a spot on the national team. The tournament is broadcast in multiple languages and watched by millions across the globe, making it a cultural and commercial juggernaut.
"""

text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=10
)
# Combine all page contents into one large string (optional)
full_text = "".join([doc.page_content for doc in docs])

# Split the full text into chunks
split_text = text_splitter.split_text(full_text)

# Print the split result
for i, chunk in enumerate(split_text[:-5]):  # printing first 5 chunks for brevity
    print(f"Chunk {i+1}:\n{chunk}\n")
