# verify_data.py
import json

print("Verifying downloaded data...")
print("=" * 40)

# Check documents
with open('data/raw/clinicaltrials_full.json', 'r', encoding='utf-8') as f:
    docs = json.load(f)
print(f"✅ Documents: {len(docs)}")
print(f"   First doc_id: {docs[0]['doc_id']}")
print(f"   First title: {docs[0]['title'][:80]}...")
print()

# Check queries
with open('data/raw/queries.json', 'r', encoding='utf-8') as f:
    queries = json.load(f)
print(f"✅ Queries: {len(queries)}")
print(f"   First query: {queries[0]['text'][:80]}...")
print()

# Check qrels
with open('data/raw/qrels.json', 'r', encoding='utf-8') as f:
    qrels = json.load(f)
print(f"✅ Qrels: {len(qrels)}")
print(f"   First qrel: {qrels[0]}")

print("=" * 40)
print("🎉 All files are valid and ready to use!")