# embedding_search.py - Semantic search using BERT embeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Same documents
documents = {
    "doc1": "Machine learning is a way to teach computers without programming them",
    "doc2": "Information retrieval helps people find relevant documents",
    "doc3": "Python is a programming language used for data science",
    "doc4": "Search engines use BM25 to rank web pages",
    "doc5": "Deep learning is a subset of machine learning",
    "doc6": "Natural language processing helps computers understand text",
    "doc7": "Artificial intelligence makes machines think like humans",
    "doc8": "Data science uses statistics and machine learning",
    "doc9": "Web search engines like Google find information online",
    "doc10": "Ranking algorithms order search results by relevance"
}

doc_ids = list(documents.keys())
doc_texts = list(documents.values())

# Load BERT model (this downloads once, takes ~1-2 minutes first time)
print("📚 Loading BERT model (first time may take 1-2 minutes)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded!")

# Create embeddings for all documents
print("📊 Creating document embeddings...")
doc_embeddings = model.encode(doc_texts)
print(f"✅ Created {len(doc_embeddings)} document embeddings")
print(f"   Each embedding has {len(doc_embeddings[0])} dimensions")

def search(query, top_k=3):
    print(f"\n🔍 Query: '{query}'")
    
    # Create query embedding
    query_embedding = model.encode([query])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
    
    # Get top k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\n📄 Top {top_k} results:")
    for idx in top_indices:
        print(f"   {doc_ids[idx]}: {similarities[idx]:.4f}")
        print(f"   Text: {documents[doc_ids[idx]][:80]}...")
        print()
    
    return [(doc_ids[i], similarities[i]) for i in top_indices]

# Test queries
print("=" * 50)
print("BERT EMBEDDING SEARCH - TESTING")
print("=" * 50)

search("machine learning", top_k=3)
search("search engines", top_k=3)
search("python programming", top_k=3)
search("artificial intelligence", top_k=3)

print("✅ Done! Your BERT embedding search is working!")