from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_community.document_loaders.text import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_core.messages import HumanMessage

load_dotenv()

loader = TextLoader("hr_policy.txt")

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)
# print("docs: ", docs)
embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

question = "i am unwell so give me leave details"

retrieved_docs = vectorstore.similarity_search(
    question,
    k=3
)

context = "\n\n".join(
    [doc.page_content for doc in retrieved_docs]
)

prompt = f"""
Answer only using the HR policy.

Context:
{context}

Question:
{question}
"""

llm = ChatOpenAI(
    model="gpt-4.1-mini"
)

response = llm.invoke(
    [HumanMessage(content=prompt)]
)

print(response.content)