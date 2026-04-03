import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

chat_history=[]

embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(
    index_name=os.environ['INDEX_NAME'], embedding=embeddings
)

chat = ChatOpenAI(verbose=True, temperature=0, model_name='gpt-4o-mini')

qa = ConversationalRetrievalChain.from_llm(
    llm=chat, chain_type='stuff', retriever=vectorstore.as_retriever()
)

res = qa.invoke({'question': 'What are the novel contributions of the paper in the realm on steering vectors?', 'chat_history': chat_history})
print(res)

history = (res['question'], res['answer'])
chat_history.append(history)

res = qa.invoke({'question': "Can you please elaborate on the second one?", 'chat_history': chat_history})
print(res)