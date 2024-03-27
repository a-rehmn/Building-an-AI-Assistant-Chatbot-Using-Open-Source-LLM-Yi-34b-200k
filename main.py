!pip install langchain
!pip install PyPDF2
!pip install huggingface_hub
!pip install sentence_transformers
!pip install faiss-cpu
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
# Getting the OpenAI API
import os
os.environ["HUGGINGFACE_API_KEY"] = "use your key"
# connect your Google Drive
from google.colab import drive
drive.mount('/content/drive')
# location of the pdf file/files.
reader = PdfReader('/content/drive/MyDrive/TR_data/TR.pdf')

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text
# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits.

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 0,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# Download embeddings from HUGGING FACE
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
db = FAISS.from_texts(texts, embeddings)

db
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

llm=HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature":0.5, "max_length":512},
    huggingfacehub_api_token="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",)
chain = load_qa_chain(llm, chain_type="stuff")

query = "who are the authors of the article?"
docs = db.similarity_search(query)
chain.run(input_documents=docs, question=query)
query = "What is the model size of GPT4all?"
docs = db.similarity_search(query)
chain.run(input_documents=docs, question=query)
query = "How was the GPT4all model trained?"
docs = db.similarity_search(query)
chain.run(input_documents=docs, question=query)
