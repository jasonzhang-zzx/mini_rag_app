import torch
from sentence_transformers import SentenceTransformer, CrossEncoder


def cos_sim(a, b):
    """calculates the cosine similarity between two vectors"""
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

def l2_distance(a, b):
    """calculates the Euclidean distance between two vectors"""
    return torch.norm(a - b)

def get_embedding(texts, model="all-MiniLM-L6-v2"):
    """
    encapsulate the embedding interface, using sentence-transformers
    """
    # load embedding model
    embedding_model = SentenceTransformer(model)
    
    embeddings = embedding_model.encode(texts)
    return [embedding for embedding in embeddings]

def reranker(query, corpus, model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=5):
    """encapsulate the reranker interface, using sentence-transformers"""
    rerank_model = CrossEncoder(model)
    ranks = rerank_model.rank(query, corpus)
    # reutrn the top_n corpus
    return [corpus[rank["corpus_id"]] for rank in ranks[:top_n]]



if __name__ == "__main__":

    test_queries = ["global conflict", "the Ukraine war"]
    embedded_vectors = get_embedding(test_queries, model="sentence-transformers/LaBSE")
    print(len(embedded_vectors[0]))
    print(embedded_vectors[0][:10])
    print(embedded_vectors[1][:10])
    print(cos_sim(torch.tensor(embedded_vectors[0]), torch.tensor(embedded_vectors[1])))

