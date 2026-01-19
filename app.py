from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
#from langchain.chains import create_retrieval_chain
#from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
# LLM
from langchain_openai import ChatOpenAI

# Prompt
from langchain_core.prompts import ChatPromptTemplate

# Embeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

# Vector DB (modern 2026 way)
import pinecone


from langchain_openai import ChatOpenAI
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


# Initialize LLM
def ask_question(retriever, llm, prompt_text, query):
    # Step 1: Retrieve documents
    docs = retriever.vectorstore.similarity_search(query, k=3)
    
    # Step 2: Combine content
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Step 3: Fill prompt
    full_prompt = prompt_text.format(context=context, input=query)
    
    # Step 4: Call LLM (returns AIMessage)
    response = llm.invoke(full_prompt)
    
    # Step 5: Return AIMessage directly
    return response


# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Prompt template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n{context}"
)

answer = ask_question(retriever, llm, system_prompt, "What is Acromegaly and gigantism?")
print(answer)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_text = msg
    print("Input:", input_text)
    
    # Call ask_question
    response = ask_question(retriever, llm, system_prompt, input_text)
    
    # If response is AIMessage, get its content
    if hasattr(response, "content"):
        response_text = response.content
    else:
        response_text = str(response)  # fallback
    
    print("Response:", response_text)
    
    # Return a string (Flask-compatible)
    return response_text



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)