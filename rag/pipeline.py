import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# These load once when the module is imported, not on every query
EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
CLIENT = chromadb.PersistentClient(path="data/chroma")
COLLECTION = CLIENT.get_or_create_collection("coursework")

SYSTEM_PROMPT = """You are a study assistant helping a university student understand their course material.
Answer ONLY using the provided context chunks below.
If the context does not contain enough information to answer, say "I could not find this in your notes."
Keep answers concise and clear."""

