import os
from langchain_core.prompts import PromptTemplate


template = PromptTemplate(
    template="""
    Please summarize the research paper titled: {paper_input} with the following specifications:
    
    Explanation style: {style_input},
    Explanation length: {length_input}

    1. Mathematical details:
        - Include relevant mathematical equations if present in the paper.
        - Explain mathematical concepts in a simple and intuitive manner.

    2. Analogies:
        - Use relatable analogies to explain complex ideas.
    If certain information is not available in the paper, respond with "Insufficient information is available."

    Make sure the result is clear and aligned with the input for consistent results.
    """,
    input_variables=['paper_input', 'style_input', 'length_input' ] , 
    validate_template= True
)


template.save('template.json')
