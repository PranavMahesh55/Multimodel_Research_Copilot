import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(
    index_name=os.environ['INDEX_NAME'], embedding=embeddings
)

chat = ChatOpenAI(verbose=True, temperature=0, model_name='gpt-4o-mini')

qa = RetrievalQA.from_chain_type(
    llm=chat, chain_type='stuff', retriever=vectorstore.as_retriever()
)

res = qa.invoke('What are steering vectors commonly used for according to the paper?')
print(res)

res = qa.invoke("What are the novel contributions of the paper in the realm on steering vectors?")
print(res)