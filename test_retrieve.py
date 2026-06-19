from rag.pipeline import retrieve

chunks = retrieve("What is the Capability Maturity Model?")
print(f"Retrieved {len(chunks)} chunks\n")
for i, (text, meta, dist) in enumerate(chunks, 1):
    print(f"--- Chunk {i} ---")
    print(f"Source: {meta.get('source')} | Loc: {meta.get('page') or meta.get('slide') or '?'} | Distance: {dist:.3f}")
    print(f"Text: {text[:250]}")
    print()