import chromadb
from chromadb.config import Settings
from extract_from_pdf import extract_text_from_pdf, split_text
from embeddings import get_embedding
import os

class MyVectorDBConnector:
    def __init__(self, collection_name: str, embedding_function):
        chroma_client = chromadb.PersistentClient(path="./chroma")

        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        self.embedding_function = embedding_function
    
    def add_documents(self, documents: list[str]):
        """add documents to the collection"""
        self.collection.add(
            embeddings=self.embedding_function(documents), 
            documents=documents, 
            ids=[f"id{i}" for i in range(len(documents))],
        )
    
    def search(self, query: str, top_k: int = 5):
        """search the collection"""
        results = self.collection.query(
            query_embeddings=self.embedding_function([query]), 
            n_results=top_k,
        )
        return results



if __name__ == "__main__":
    paragaphs = []
    for file in os.listdir("./literatures"):
        if file.endswith(".pdf"):
            filepath = os.path.join("./literatures", file)
            paragaph = extract_text_from_pdf(filepath, min_line_length=8)
            paragaphs.extend(paragaph)

    chunks = split_text(paragaphs)

    vec_db = MyVectorDBConnector("defult", embedding_function=get_embedding)
    vec_db.add_documents(chunks)
    print("A new vector database is created.")


