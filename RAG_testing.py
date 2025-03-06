import RAG  # Import your Flask app module
import os

# Test extracting text from a sample PDF file
def test_extract_text():
    pdf_path = "sample.pdf"  # Replace with an actual PDF file path
    if os.path.exists(pdf_path):
        extracted_text = RAG.extract_text_from_pdf(pdf_path)
        print("Extracted Text:", extracted_text[:500])  # Print first 500 chars for preview
    else:
        print("PDF file not found!")

# Test the RAG processing function
def test_rag():
    sample_text = "This is a test document. It contains useful information."
    response = RAG.process_with_rag(sample_text)
    print("RAG Response:", response)

# Run the tests
if __name__ == "__main__":
    test = 'nAPI'
    if test == 'API':
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        print(OPENAI_API_KEY)
        folder_path = './uploads'
        folder_path_list = os.listdir(folder_path)
        print(folder_path_list)
        text = RAG.extract_text_from_pdf(folder_path)
        response = RAG.RAG_process()
        print(response)
    else:
        import fitz 
        from langchain.vectorstores import Chroma
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.llms import OpenAI
        from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
        from langchain.document_loaders import TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain import embeddings
        from langchain.chains import RetrievalQA

        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        folder_path = './uploads'
        text = RAG.extract_text_from_pdf(folder_path)

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
        retriever = vectordb.as_retriever(search_kwargs={"k":10})
        
        query = 'Summarize concepts relating to job search.'

        docs = retriever.get_relevant_documents(query)

        llm = OpenAI()
        qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                            chain_type="stuff",
                                            retriever=retriever,
                                            return_source_documents=True)

        llm_response = qa_chain(query)
        response = llm_response['result']
        print(response)
