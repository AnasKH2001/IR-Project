# vsm_search.py - TF-IDF Vector Space Model search engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Same 10 fake documents
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

# Step 2: Prepare documents as list
doc_ids = list(documents.keys())
doc_texts = list(documents.values())

# Step 3: Build TF-IDF Vectorizer
print("📚 Building TF-IDF index...")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(doc_texts)
print(f"✅ Indexed {len(doc_ids)} documents")
print(f"   Vocabulary size: {len(vectorizer.get_feature_names_out())} words")

# Step 4: Search function
def search(query, top_k=3):
    print(f"\n🔍 Query: '{query}'")
    
    # Convert query to TF-IDF vector
    query_vec = vectorizer.transform([query])
    
    # Calculate cosine similarity between query and all documents
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Show results
    print(f"\n📄 Top {top_k} results:")
    for idx in top_indices:
        if similarities[idx] > 0:
            print(f"   {doc_ids[idx]}: {similarities[idx]:.4f}")
            print(f"   Text: {documents[doc_ids[idx]][:80]}...")
            print()
    
    return [(doc_ids[i], similarities[i]) for i in top_indices if similarities[i] > 0]

# Step 5: Test the same queries as BM25
print("=" * 50)
print("TF-IDF (VSM) SEARCH ENGINE - TESTING")
print("=" * 50)

# Test 1
search("machine learning", top_k=3)

# Test 2
search("search engines", top_k=3)

# Test 3
search("python programming", top_k=3)

print("✅ Done! Your TF-IDF search engine is working!")