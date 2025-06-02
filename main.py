from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (for frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load your content
loader = TextLoader("docs/sample.txt", encoding="utf-8")
documents = loader.load()

# Split into manageable chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Create embeddings and vector store
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma.from_documents(docs, embedding=embedding, persist_directory="db")
retriever = vectorstore.as_retriever()

# Restrictive custom prompt
template = """
You are an enthusiastic Hawaiian museum tour guide. Answer the question using only the information provided in the context below.
If the answer cannot be found in the context, say "I'm not sure about that one, but great question!"

Context:
{context}

Question:
{question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# Set up LangChain QA system with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_api_key),
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# Define question format
class Question(BaseModel):
    question: str

# Create an endpoint for frontend requests
@app.post("/ask")
async def ask(payload: Question):
    response = qa_chain.run(payload.question)
    return {"answer": response}
