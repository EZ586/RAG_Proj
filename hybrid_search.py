import os

import weaviate
from weaviate.classes.init import Auth

import fitz 
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever


# Best practice: store your credentials in environment variables
# wcd_url = os.environ["WCD_URL"]
# wcd_api_key = os.environ["WCD_API_KEY"]
wcd_url = 'https://wgdoph0qxingj5watukg.c0.us-east1.gcp.weaviate.cloud'
wcd_api_key = '3VyrchKzfpliclX6BPCVSRHLCF0I4CvJRE0t'

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,
    auth_credentials=Auth.api_key(wcd_api_key),
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists

# OpenAI API Key (set this in environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define Weaviate Index Name
INDEX_NAME = "LangChain"

# Set up OpenAI Embeddings
embedding = OpenAIEmbeddings()

# Function to check allowed file type
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract text from PDFs
def extract_text_from_pdf(folder_path):
    text = ""
    for path in os.listdir(folder_path):
        path = os.path.join(folder_path, path)
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    return text

# Function to create/update vector database in Weaviate
def create_vectordb():
    # Load and process PDFs
    folder_path = "./uploads"
    loader = DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    text_chunks = text_splitter.split_documents(documents)

    # Upload documents into Weaviate
    for chunk in text_chunks:
        doc_text = chunk.page_content
        doc_embedding = embedding.embed_query(doc_text)  # Generate vector

        client.data_object.create(
            data_object={"text": doc_text},
            class_name=INDEX_NAME,
            vector=doc_embedding
        )

    print("Weaviate vector database updated with new documents.")

# Query processing using Weaviate Hybrid Search
@app.route("/query", methods=["POST"])
def process_query():
    query = request.form.get("query")
    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    retriever = WeaviateHybridSearchRetriever(
        alpha=0.5,  # Equal weighting between keyword (BM25) and semantic search
        client=client,
        index_name=INDEX_NAME,
        text_key="text"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    llm_response = qa_chain(query)
    response = llm_response["result"]

    return jsonify({"message": response}), 200

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

        # Update Weaviate vector database
        create_vectordb()

        # Return list of uploaded files
        files_list = os.listdir(UPLOAD_FOLDER)
        return jsonify({
            "message": "File processed successfully",
            "response": "PDF RECEIVED!",
            "files": files_list
        })
    else:
        return jsonify({"error": "Invalid file type"}), 400

# Endpoint to list uploaded files
@app.route("/")
def files_list():
    files_list = os.listdir(UPLOAD_FOLDER)
    return jsonify({"files": files_list})

if __name__ == "__main__":
    app.run()
