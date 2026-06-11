# compare_models.py - Compare BM25 vs TF-IDF side by side
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
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

# Build BM25
print("Building BM25...")
tokenized_docs = [text.lower().split() for text in doc_texts]
bm25 = BM25Okapi(tokenized_docs)

# Build TF-IDF
print("Building TF-IDF...")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(doc_texts)

def search_bm25(query, top_k=3):
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(doc_ids[i], scores[i]) for i in top_indices if scores[i] > 0]

def search_tfidf(query, top_k=3):
    query_vec = vectorizer.transform([query.lower()])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(doc_ids[i], similarities[i]) for i in top_indices if similarities[i] > 0]

# Test queries
test_queries = ["machine learning", "search engines", "python", "artificial intelligence"]

print("\n" + "=" * 60)
print("COMPARISON: BM25 vs TF-IDF")
print("=" * 60)

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    print("-" * 40)
    
    bm25_results = search_bm25(query, 3)
    tfidf_results = search_tfidf(query, 3)
    
    print("BM25 Results:")
    for doc_id, score in bm25_results:
        print(f"   {doc_id}: {score:.4f}")
    
    print("\nTF-IDF Results:")
    for doc_id, score in tfidf_results:
        print(f"   {doc_id}: {score:.4f}")
    print()

print("✅ Comparison complete!")