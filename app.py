from flask import Flask, request, jsonify, render_template
from src.helper import download_model
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains  import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app=Flask(__name__)


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


embedding = download_model() 


index_name = "medical-chatbot"

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding)

docsearch=PineconeVectorStore.from_existing_index(index_name=index_name,embedding=embedding)


PROMPT=PromptTemplate(template=prompt_template,input_variables=["context","question"])
chain_type_kwargs={"prompt":PROMPT}


llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})



qa=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k":2}),
    chain_type_kwargs=chain_type_kwargs
)

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/get', methods=['POST','GET'])
def chat():
    msg=request.form['msg']
    input=msg
    print(input)
    result=qa.invoke({"query":input})
    print("Response: ",result["result"])
    return str(result["result"])


if __name__ == "__main__":
    app.run(debug=True)