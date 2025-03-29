from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from typing import TypedDict

"""
-since huggingface model is not giving structured output this code won't work.
- when you change the model to open ai or any model that is Able to generate syructure output 
- this code will work at that tieme
"""

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="distilbert-base-uncased-finetuned-sst-2-english",
    task="text-classification",
)

sentimnet_model = ChatHuggingFace(llm=llm)

llm2 = HuggingFaceEndpoint(
    repo_id="facebook/bart-large-cnn",
    task="summarization",
)

summary_model = ChatHuggingFace(llm=llm)


class review(TypedDict):
    
    sentiment : str
    score     : float

class review2(TypedDict):
    summary  : str
    lenth    : int
sentinmet = sentimnet_model.with_structured_output(review)
summary = summary_model.invoke(review2)
sentinmet_result = sentinmet.invoke('I recently acquired a pair of shoes with excellent quality. The craftsmanship is impeccable, and the materials used are top-notch. The comfort level is outstanding, making them a fantastic investment for anyone seeking durable and stylish footwear.')
summary_result = summary_model.invoke('I recently acquired a pair of shoes with excellent quality. The craftsmanship is impeccable, and the materials used are top-notch. The comfort level is outstanding, making them a fantastic investment for anyone seeking durable and stylish footwear.')
print(sentinmet_result)
print (summary_result)
