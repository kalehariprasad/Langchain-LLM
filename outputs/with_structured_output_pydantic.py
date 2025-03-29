from pydantic import BaseModel , Field
from typing import Optional,Literal
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

"""
-since huggingface model is not giving structured output this code won't work.
- when you change the model to open ai or any model that is Able to generate syructure output 
- this code will work at that tieme
"""


llm = HuggingFaceEndpoint(
    repo_id="facebook/bart-large-cnn",
    task="summarization",
)
model = ChatHuggingFace(llm=llm)


class review(BaseModel):
    key_features :list[str]=Field(deprecated='ectract all key features that are used in the review')
    summary      : str = Field(description='genarate the summary of a review')
    sentiment    :Literal['positive','negitive']=Field(description='generate snetiment of the review in poitive or negitive')
    pros         : list[str]=Field(default=None,description='hilight all pros that are mentioned in review')
    cons         :list[str]=Field(default=None,description='hiulight all cons that aew mentioned in the review')

sentinmet = model.with_structured_output(review)
result = model.invoke('I recently acquired a pair of shoes with excellent quality. The craftsmanship is impeccable, and the materials used are top-notch. The comfort level is outstanding, making them a fantastic investment for anyone seeking durable and stylish footwear.')
print(result)

