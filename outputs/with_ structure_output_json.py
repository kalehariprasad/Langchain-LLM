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


json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

sentinmet = model.with_structured_output(json_schema)
result = model.invoke('I recently acquired a pair of shoes with excellent quality. The craftsmanship is impeccable, and the materials used are top-notch. The comfort level is outstanding, making them a fantastic investment for anyone seeking durable and stylish footwear.')
print(result)

