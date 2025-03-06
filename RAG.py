import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI

import fitz 
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import embeddings
from langchain.chains import RetrievalQA


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists

# OpenAI API Key (set this in environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# set up vector database
# Define the persistence directory
persist_directory = "new_db"

# Check if the directory exists
if os.path.exists(persist_directory) and os.listdir(persist_directory):
    print("Loading existing vector database...")
    embedding = OpenAIEmbeddings()  # Required for loading
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
else:
    print("No existing database found. Creating new vector store...")

    # Load and process documents
    loader = DirectoryLoader("./uploads", glob="*.pdf")  # Load PDFs from folder
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    text_chunks = text_splitter.split_documents(documents)

    # Create embeddings and vector database
    embedding = OpenAIEmbeddings()  # This incurs OpenAI API costs
    vectordb = Chroma.from_documents(documents=text_chunks, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()
    print("Vector database created and saved.")


# Function to check allowed file type
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract text from PDF
def extract_text_from_pdf(folder_path):
    text = ""
    for path in os.listdir(folder_path):
        path = os.path.join(folder_path, path)
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    return text

# Process Query
@app.route("/query", methods=["POST"])
def process_query():
    query = request.form.get("query")  # Using .get() to avoid KeyError
    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400
    retriever = vectordb.as_retriever(search_kwargs={"k":10})
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                        chain_type="stuff",
                                        retriever=retriever,
                                        return_source_documents=True)

    llm_response = qa_chain(query)
    response = llm_response['result']

    return jsonify({"message": response}), 200

# Function to process text with RAG pipeline
def create_vectordb():
    folder_path = './uploads'
    text = extract_text_from_pdf(folder_path)

    loader = DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyMuPDFLoader)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 400)
    text = text_splitter.split_documents(document)

    persist_directory = 'new_db'

    embedding = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(documents=text,
                                embedding=embedding,
                                persist_directory=persist_directory)
    
    vectordb.persist()
    vectordb = None

    vectordb = Chroma(persist_directory=persist_directory,
                embedding_function=embedding)

# Route to handle PDF file uploads
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        # update vector database
        create_vectordb()
        # Get the list of uploaded files
        files_list = os.listdir(UPLOAD_FOLDER)  # List all files in upload directory

        return jsonify({"message": "File processed successfully", 
                        "response": "PDF RECEIVED!",
                        "files": files_list})  # Return file list to frontend
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route('/')
def files_list():
    files_list = os.listdir(UPLOAD_FOLDER)  # List all files in upload directory
    return jsonify({"files": files_list})

if __name__ == '__main__':
    app.run()