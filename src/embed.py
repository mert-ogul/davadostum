"""
Create SBERT embeddings for each decision paragraph and build FAISS index.
Run:
    python -m legalrag.embed
    python -m legalrag.embed --force  # Force re-embedding
"""

import argparse
from pathlib import Path
import sqlite3
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from .settings import load_config
from .utils import ensure_dirs


def paragraphs(text: str, max_len: int = 2000):
    """Yield meaningful legal document chunks from the given text."""
    import re
    
    # Clean the text first
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split by major legal document sections
    sections = re.split(r'(SUÇ\s*:|HÜKÜM\s*:|GEREKÇE\s*:|KARAR\s*:|SONUÇ\s*:|TEMYİZ\s*:)', text)
    
    if len(sections) <= 1:
        # If no major sections, split by sentences
        sentences = re.split(r'[.!?]+', text)
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_len:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    yield current_chunk.strip()
                current_chunk = sentence + ". "
        if current_chunk.strip():
            yield current_chunk.strip()
        return
    
    # Process sections
    current_chunk = ""
    for i, section in enumerate(sections):
        if i % 2 == 0:  # Content
            if len(current_chunk + section) <= max_len:
                current_chunk += section
            else:
                if current_chunk.strip():
                    yield current_chunk.strip()
                current_chunk = section
        else:  # Section header
            if len(current_chunk + section) <= max_len:
                current_chunk += section
            else:
                if current_chunk.strip():
                    yield current_chunk.strip()
                current_chunk = section
    
    # Handle remaining text
    if current_chunk.strip():
        yield current_chunk.strip()


def get_db_decision_count(db_path: str) -> int:
    """Get total number of decisions in database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM decisions")
    count = cur.fetchone()[0]
    conn.close()
    return count


def get_meta_decision_count(meta_path: str) -> int:
    """Get number of unique decisions in meta.json."""
    if not Path(meta_path).exists():
        return 0
    
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        # Count unique decision IDs
        unique_ids = set(item['id'] for item in meta)
        return len(unique_ids)
    except (json.JSONDecodeError, KeyError):
        return 0


def get_model():
    """Load a high-quality multilingual model for Turkish legal embeddings."""
    # Using a more advanced multilingual model for better Turkish legal text understanding
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


def embed_texts(texts):
    model = get_model()
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


def main():
    """Main function to embed all decisions."""
    # Load decisions from database
    conn = sqlite3.connect("data/decisions.sqlite")
    cur = conn.cursor()
    cur.execute("SELECT id, raw_text FROM decisions")
    db_decisions = cur.fetchall()
    conn.close()
    
    print(f"Found {len(db_decisions)} decisions in database")
    
    # Load model
    model = get_model()
    
    # Process decisions with deduplication
    all_paragraphs = []
    seen_decisions = set()  # Track processed decision IDs
    
    for decision_id, raw_text in tqdm(db_decisions, desc="Chunking texts"):
        # Skip if already processed
        if decision_id in seen_decisions:
            continue
        seen_decisions.add(decision_id)
        
        chunks = paragraphs(raw_text)
        for chunk in chunks:
            all_paragraphs.append({"id": decision_id, "snippet": chunk})
    
    print(f"Created {len(all_paragraphs)} unique chunks from {len(seen_decisions)} decisions")
    
    # Batch embedding
    batch_size = 32
    embeddings = []
    meta_data = []
    
    for i in tqdm(range(0, len(all_paragraphs), batch_size), desc="Embedding chunks"):
        batch_paragraphs = all_paragraphs[i:i + batch_size]
        batch_texts = [p["snippet"] for p in batch_paragraphs]
        batch_embeddings = model.encode(batch_texts, batch_size=8, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
        meta_data.extend(batch_paragraphs)
    
    # Save embeddings and metadata
    embeddings = np.array(embeddings)
    np.save("data/embeddings.npy", embeddings)
    
    with open("data/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, "data/faiss.index")
    
    print(f"Saved {len(embeddings)} embeddings with dimension {dimension}")
    print("Embedding process completed successfully!")


if __name__ == "__main__":
    main()
