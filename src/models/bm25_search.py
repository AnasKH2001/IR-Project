# bm25_search.py - Simple BM25 search engine with fake data
from rank_bm25 import BM25Okapi

# Step 1: Create 10 fake documents (our "dataset")
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

# Step 2: Preprocess each document (lowercase + split into words)
tokenized_docs = []
doc_ids = []

for doc_id, text in documents.items():
    # Simple preprocessing: lowercase and split
    tokens = text.lower().split()
    tokenized_docs.append(tokens)
    doc_ids.append(doc_id)

# Step 3: Build BM25 index
print("📚 Building BM25 index...")
bm25 = BM25Okapi(tokenized_docs)
print(f"✅ Indexed {len(doc_ids)} documents")

# Step 4: Search function
def search(query, top_k=3):
    print(f"\n🔍 Query: '{query}'")
    
    # Preprocess query the same way
    query_tokens = query.lower().split()
    
    # Get scores
    scores = bm25.get_scores(query_tokens)
    
    # Get top results
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    # Show results
    print(f"\n📄 Top {top_k} results:")
    for idx in top_indices:
        if scores[idx] > 0:
            print(f"   {doc_ids[idx]}: {scores[idx]:.4f}")
            print(f"   Text: {documents[doc_ids[idx]][:80]}...")
            print()
    
    return [(doc_ids[i], scores[i]) for i in top_indices if scores[i] > 0]

# Step 5: Test some queries
print("=" * 50)
print("BM25 SEARCH ENGINE - TESTING")
print("=" * 50)

# Test 1
search("machine learning", top_k=3)

# Test 2
search("search engines", top_k=3)

# Test 3
search("python programming", top_k=3)

print("✅ Done! Your BM25 search engine is working!")