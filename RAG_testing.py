import RAG  # Import your Flask app module
import os


# Run the tests
if __name__ == "__main__":
    test = 'test2'
    if test == 'API':
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        print(OPENAI_API_KEY)
        folder_path = './uploads'
        folder_path_list = os.listdir(folder_path)
        print(folder_path_list)
        text = RAG.extract_text_from_pdf(folder_path)
        response = RAG.RAG_process()
        print(response)
    elif test == 'nAPI':
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
    elif test == 'test2':
        import os
        from langchain.retrievers import BM25Retriever, EnsembleRetriever
        from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.vectorstores import FAISS
        from langchain.vectorstores import Chroma
        from langchain.embeddings.openai import OpenAIEmbeddings
        # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        folder_path = './uploads'

        loader = DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyMuPDFLoader)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 400)
        doc_list = text_splitter.split_documents(document)
        text_list = [text.page_content for text in doc_list]
        # print(20 * '-')
        # print(doc_list[0].page_content)
        # print(doc_list[0].metadata.keys())
        # print(doc_list[0].metadata['title'])
        # print(20 * '-')
        # for text in text_list:
        #     print(text)
        #     print(20 * '@')

        # intialize BM25 Retriever - Sparce Retriever
        bm25_retriever = BM25Retriever.from_texts(text_list)
        bm25_retriever.k = 2
        rel_doc = bm25_retriever.get_relevant_documents("What is the weaknesses relating to the paper on influenza?")
        # for doc in rel_doc:
        #     print(20 * '^')
        #     print(doc.page_content)
        #     print(20 * '@')
        # print(20 * '%')

        # intialize FAISS Retriever - Dense retriever 
        persist_directory = "new_db"
        vectordb = None
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            print("Loading existing vector database...")
            embedding = OpenAIEmbeddings()  # Required for loading
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        retriever = vectordb.as_retriever(search_kwargs={"k":2})
        # embedding = OpenAIEmbeddings()
        # faiss_vectorstore = FAISS.from_texts(text_list, embedding)
        # faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})
        fais_rel_doc = retriever.get_relevant_documents("What assessments were made about Chat-GPT?")
        # for doc in fais_rel_doc:
        #     print(20 * '^')
        #     print(doc.page_content)
        #     print(20 * '@')

        # Ensemble Retriever
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever],
                                       weights=[0.5, 0.5])
        docs = ensemble_retriever.get_relevant_documents("What are the 3 main parts of IEEE citation?")
        print('\n' + 20 * '$')
        for doc in docs:
            print(20 * '^')
            print(doc.page_content)
            print(20 * '@')



