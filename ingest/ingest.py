from pathlib import Path
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
from ingest.loaders import load_any
from ingest.chunker import chunk_text

def ingest(raw_dir: str ="data/raw", db_str: str = "data/chroma"):
    # Load embedding model on the first run then cache it for later
    print("Loading embedding model...")
    emedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Connect to ChromaDB (will create DB if it doesn't exist)
    client = chromadb.PersistentClient(path=db_str)
    collection = client.get_or_create_collection("documents")

    # Find all files in the raw directory
    files = [
        p for p in Path(raw_dir).rglob("*") 
        if p.is_file() and not p.name.startswith(".")  # skip hidden files
    ]
    print(f"Found {len(files)} files to ingest.")

    # Load, chunk, and embed each file, then add to ChromaDB
    docs, metas, ids =[], [], []
    counter = 0
    for path in tqdm(files, desc="Loading and chunking files"):
        for text, meta in load_any(path):
            for chunk in chunk_text(text):
                docs.append(chunk)
                metas.append(meta)
                ids.append(f"{path.name}_{counter}")
                counter += 1
    
    print(f"Total chunks created: {len(docs)}")

    # embed all chunks
    print("Embedding chunks...")
    embeddings = emedder.encode(
        docs,
        show_progress_bar=True,
        batch_size=32,
    ).tolist()

    # Add to ChromaDB
    print("Adding to ChromaDB...")
    BATCH_SIZE = 1000
    for i in range(0, len(docs), BATCH_SIZE):
        collection.add(
            documents=docs[i:i+BATCH_SIZE],
            embeddings=embeddings[i:i+BATCH_SIZE],
            metadatas=metas[i:i+BATCH_SIZE],
            ids=ids[i:i+BATCH_SIZE],
        )
    
    print(f"Done, {collection.count()} total chunks in ChromaDB.")

if __name__ == "__main__":
    ingest()