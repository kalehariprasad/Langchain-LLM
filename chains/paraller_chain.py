from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

#RunnableParallel parameter is used for runninfg parrele chains at a time 
load_dotenv()


llm= HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
     task="text-generation",
 )
model = ChatHuggingFace(llm=llm)

llm2 =HuggingFaceEndpoint(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation'
)

model2=ChatHuggingFace(llm = llm2)

parser=StrOutputParser()

prompt1 = PromptTemplate(
    template= 'genarate the simple and short nots based on the given {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='geanarate 5 Quiz Question and answer from the give {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(

    template= ' merge both {notes} and {quiz} as single document',
    input_variables= ['notes','quiz']
)



parallel_chain = RunnableParallel({
    'notes':prompt1|model|parser,
    'quiz':prompt2|model2|parser
})

merge_cahin= prompt3|model |parser

chain = parallel_chain|merge_cahin

text = """
The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares (see below). This algorithm requires the number of clusters to be specified. It scales well to large numbers of samples and has been used across a large range of application areas in many different fields.

The k-means algorithm divides a set of 
 samples 
 into 
 disjoint clusters 
, each described by the mean 
 of the samples in the cluster. The means are commonly called the cluster “centroids”; note that they are not, in general, points from 
, although they live in the same space.

The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion:

 
 
Inertia can be recognized as a measure of how internally coherent clusters are. It suffers from various drawbacks:

Inertia makes the assumption that clusters are convex and isotropic, which is not always the case. It responds poorly to elongated clusters, or manifolds with irregular shapes.

Inertia is not a normalized metric: we just know that lower values are better and zero is optimal. But in very high-dimensional spaces, Euclidean distances tend to become inflated (this is an instance of the so-called “curse of dimensionality”). Running a dimensionality reduction algorithm such as Principal component analysis (PCA) prior to k-means clustering can alleviate this problem and speed up the computations.

K-means is often referred to as Lloyd’s algorithm. In basic terms, the algorithm has three steps. The first step chooses the initial centroids, with the most basic method being to choose 
 samples from the dataset 
. After initialization, K-means consists of looping between the two other steps. The first step assigns each sample to its nearest centroid. The second step creates new centroids by taking the mean value of all of the samples assigned to each previous centroid. The difference between the old and the new centroids are computed and the algorithm repeats these last two steps until this value is less than a threshold. In other words, it repeats until the centroids do not move significantly.
"""


result = chain.invoke({'text':text})
print(result)

chain.get_graph().print_ascii()