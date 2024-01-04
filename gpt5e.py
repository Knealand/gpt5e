import os
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify, abort
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

load_dotenv()

VALID_API_KEY = os.environ.get("VALID_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
VDB_PATH = "5e_books_index"
RAW_DB_PATH = "books"
############################
#          GPT-5e          #
############################

vectorstore = None
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
try:
    if os.path.isdir(VDB_PATH) and len(os.listdir(VDB_PATH)) > 0:
        vectorstore = FAISS.load_local(VDB_PATH, embeddings)
    else:
        loader = DirectoryLoader(RAW_DB_PATH, glob="**/*.md", show_progress=True)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=500,
        )

        documents = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local("5e_books_index")
except:
    print("Error could not load local VDB or raw files")


############################
#            API           #
############################
app = Flask(__name__)

def require_apikey(view_function):
    def decorated_function(*args, **kwargs):
        if request.headers.get('x-api-key') and request.headers.get('x-api-key') == VALID_API_KEY:
            return view_function(*args, **kwargs)
        else:
            abort(401)  # Unauthorized access
    return decorated_function


@app.route('/gtp5e/question', methods=['POST'])
@require_apikey
def question():
    content_type = request.headers.get('Content-Type')
    prompt = None
    if content_type == 'application/json':
        try:
            body_json = request.json
            prompt = body_json['prompt']
            prompt_template = """ """

            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            index = VectorStoreIndexWrapper(vectorstore=vectorstore)

            #chain_type_kwargs = {"prompt": PROMPT}
            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
            qa = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(model_name="gpt-4", temperature=0.1, openai_api_key=OPENAI_API_KEY),
                memory=memory,
                retriever=index.vectorstore.as_retriever(search_kwargs={"k": 3}),
                max_tokens_limit=12000,
                combine_docs_chain_kwargs={'prompt': PROMPT}
            )

            chat_history = []
            response = qa({"question": prompt, "chat_history": chat_history})
            #chat_history.append((prompt, response["answer"]))
            return jsonify({"response": response['answer']}), 200
        except:
            return jsonify({"Error": "Error processing Prompt"}), 500
        
    else:
        return jsonify({"Error": "Bad contnent type."}), 400


if __name__ == '__main__':
    app.run(debug=True)