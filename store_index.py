import os
from src.helper import load_pdf,text_splitter,download_model
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()



extracted_data = load_pdf("data/")
text_chunks = text_splitter(extracted_data)
embedding = download_model()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


index_name = "medical-chatbot"

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding)


vectorstore_from_docs = PineconeVectorStore.from_documents(
        text_chunks,
        index_name=index_name,
        embedding=embedding
    )


