from langchain.document_loaders import TextLoader,DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

data_path='data'
db_faiss_path='vectorstore/db_faiss'



# creating vector db

def create_vector_database():
    loader=DirectoryLoader(
        data_path,
        loader_cls=TextLoader,
        recursive=True,
        show_progress=True,
        use_multithreading=True,
        max_concurrency=8
    )
    
    raw_docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    ) 
    
    text=text_splitter.split_documents(raw_docs)
    
    embeddings=HuggingFaceEmbeddings(
        model_name='microsoft/unixcoder-base',
        model_kwargs={'device': 'cpu'}
    )
    
    db=FAISS.from_documents(text,embeddings)
    db.save_local(db_faiss_path)
    
if __name__=='__main__':
    create_vector_database()