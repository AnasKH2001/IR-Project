# src/utils/download_data.py
import ir_datasets
import json
import os

def download_full_dataset(dataset_name, output_file):
    """Download ALL documents from a dataset"""
    print(f"\n📥 Downloading ALL documents from {dataset_name}...")
    
    dataset = ir_datasets.load(dataset_name)
    
    documents = []
    
    for i, doc in enumerate(dataset.docs_iter()):
        # Combine all fields into one searchable text
        combined_text = f"{doc.title} {doc.condition} {doc.summary} {doc.detailed_description} {doc.eligibility}"
        
        documents.append({
            'doc_id': doc.doc_id,
            'text': combined_text,
            'title': doc.title,
            'condition': doc.condition
        })
        
        if (i + 1) % 50000 == 0:
            print(f"   Processed {i+1} documents...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved {len(documents)} documents to {output_file}")
    return documents

def download_queries_and_qrels(dataset_name, output_dir):
    """Download queries and qrels for evaluation"""
    print(f"\n📥 Downloading queries and qrels from {dataset_name}...")
    
    dataset = ir_datasets.load(dataset_name)
    
    # Save queries
    queries = []
    for query in dataset.queries_iter():
        # Combine all query fields into one text string
        query_text = f"{query.disease} {query.gene} {query.demographic} {query.other}".strip()
        queries.append({
            'query_id': query.query_id,
            'text': query_text
        })
    
    with open(f"{output_dir}/queries.json", 'w', encoding='utf-8') as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    
    # Save qrels
    qrels = []
    for qrel in dataset.qrels_iter():
        qrels.append({
            'query_id': qrel.query_id,
            'doc_id': qrel.doc_id,
            'relevance': qrel.relevance
        })
    
    with open(f"{output_dir}/qrels.json", 'w', encoding='utf-8') as f:
        json.dump(qrels, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved {len(queries)} queries and {len(qrels)} qrels")
    return queries, qrels

# Create folders
os.makedirs('data/raw', exist_ok=True)

print("=" * 60)
print("Downloading Clinical Trials 2017 Dataset")
print("=" * 60)

# Download all documents (241,006)
download_full_dataset(
    dataset_name="clinicaltrials/2017/trec-pm-2017",
    output_file="data/raw/clinicaltrials_full.json"
)

# Download queries and qrels
print("\n📋 Downloading queries and qrels...")
download_queries_and_qrels("clinicaltrials/2017/trec-pm-2017", "data/raw")

print("\n" + "=" * 60)
print("🎉 Clinical Trials 2017 download complete!")
print("=" * 60)
print("\nFiles saved:")
print("  - data/raw/clinicaltrials_full.json (241,006 documents)")
print("  - data/raw/queries.json (30 test queries)")
print("  - data/raw/qrels.json (13,019 relevance judgments)")