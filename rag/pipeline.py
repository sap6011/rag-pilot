import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# These load once when the module is imported, not on every query
EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
CLIENT = chromadb.PersistentClient(path="data/chroma")
COLLECTION = CLIENT.get_or_create_collection("documents")

SYSTEM_PROMPT = """You are a study assistant helping a university student understand their course material.
Answer ONLY using the provided context chunks below.
If the context does not contain enough information to answer, say "I could not find this in your notes."
Keep answers concise and clear."""

def retrieve(query: str, k: int = 5):
    """Embed the query and find the k most similar chunks in Chroma."""
    query_embedding = EMBEDDER.encode([query])[0].tolist()
    results = COLLECTION.query(
        query_embeddings=query_embedding,
        n_results=k,
    )
    return list(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ))

def format_context(chunks):
    """Format retrieved chunks into a readable context string for the LLM."""
    parts = []
    for i, (text, meta, _) in enumerate(chunks, 1):
        source = meta.get("source", "unknown")
        loc = meta.get("page") or meta.get("slide") or meta.get("cell") or "?"
        parts.append(f"[{i}] {source} (page/slide {loc}): \n{text}")

    return "\n\n".join(parts)

def answer_query(query: str, k: int = 5):
    """Full RAG pipeline: retrieve -> format -> generate -> return."""
    # Step 1: retrieve relevant chunks
    chunks = retrieve(query, k)

    # Step 2: format context for the LLM
    context = format_context(chunks)

    # Step 3: build the prompt for the LLM
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    # Step 4: generate answer using Ollama
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],   
    )

    return {
        "answer": response["message"]["content"],
        "sources": [
            {
                "source": meta.get("source"),
                "loc": meta.get("page") or meta.get("slide") or meta.get("cell"),
                "similarity": round(1-dist, 3),
            }
            for _, meta, dist in chunks

        ],


    }

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "What is the Capability Maturity Model?"
    result = answer_query(query)
    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== SOURCES ===")
    for s in result["sources"]:
        print(f"  {s['source']} (loc {s['loc']}) — similarity {s['similarity']}")
