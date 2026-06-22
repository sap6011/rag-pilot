"""
Eval harness for the rag-pilot pipeline.
Runs all test cases in testset.jsonl, computes retrieval and generation metrics,
and prints a per-category breakdown.
"""
import json
from collections import defaultdict
from pathlib import Path
from rag.pipeline import retrieve, answer_query

REFUSAL_PHRASES = [
    "could not find",
    "couldn't find",
    "not in your notes",
    "not contain",
    "no information",
    "cannot answer",
    "don't have",
]

def load_testset(path: str = "eval/testset.jsonl"):
    """Load test cases from JSONL file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
    
def chunk_location(meta: dict) -> int | None:
    """Extract the page/slide/cell number from a chunk's metadata."""
    return meta.get("page") or meta.get("slide") or meta.get("cell")

def evaluate_retrieval(chunks, expected_locations: list[int]):
    """
    Given retrieved chunks and expected page numbers, return:
      - hit:  did at least one retrieved chunk match an expected location?
      - rank: 1-based position of the first matching chunk (None if no hit)
    """
    if not expected_locations:
        # out-of-scope: no expected location
        return {"hit": None, "rank": None}
    
    for i, (_, meta, _) in enumerate(chunks, 1):
        loc = chunk_location(meta)
        if loc in expected_locations:
            return {"hit": True, "rank": i}
    return {"hit": False, "rank": None}

def evaluate_generation(answer_text: str, ideal_keywords: list[str], is_out_of_scope: bool):
    """
    For in-scope questions: fraction of expected keywords present (case-insensitive).
    For out-of-scope questions: did the answer contain a refusal phrase?
    """
    answer_lower = answer_text.lower()

    if is_out_of_scope:
        refused = any(phrase in answer_lower for phrase in REFUSAL_PHRASES)
        return {"refused": refused, "keyword_coverage": None}
    
    if not ideal_keywords:
        return {"refused": None, "keyword_coverage": None}
    
    hits = sum(1 for kw in ideal_keywords if kw.lower() in answer_lower)
    coverage = hits / len(ideal_keywords)

    return {"refused": None, "keyword_coverage": coverage}

def run_evaluation(testset_path: str = "eval/testset.jsonl", k=5):
    """Run evaluation on the testset and print results."""
    testset = load_testset(testset_path)
    print(f"Running eval on {len(testset)} test cases (k={k})...\n")

    results = []

    for i, case in enumerate(testset, 1):
        question = case["question"]
        expected_locs = case.get("expected_locations", [])
        ideal_kws = case.get("ideal_answer_keywords", [])
        category = case.get("category", "uncategorized")
        is_oos = category == "out_of_scope"

        print(f"[{i}/{len(testset)}] {category}: {question[:60]}...")

        # Retrieve
        chunks = retrieve(question, k=k)
        retrieval = evaluate_retrieval(chunks, expected_locs)

        # Generate
        result = answer_query(question, k=k)
        generation = evaluate_generation(result["answer"], ideal_kws, is_oos)

        results.append({
            "question": question,
            "category": category,
            "retrieval": retrieval,
            "generation": generation,
        })
    print("\n" + "=" * 60)
    print("RESULTS BY CATEGORY")
    print("=" * 60)

    # Aggregate results by category
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    for cat, items in sorted(by_category.items()):
        n = len(items)
        print(f"\n[{cat}] ({n} questions)")

        if cat == "out_of_scope":
            refused = sum(1 for r in items if r["generation"]["refused"])
            print(f"  Refusal accuracy: {refused}/{n} ({100*refused/n:.0f}%)")

        #Recall@k
        hits = sum(1 for r in items if r["retrieval"]["hit"])
        print(f"  Recall@{k}: {hits}/{n} ({100*hits/n:.0f}%)")

        # MRR (Mean Reciprocal Rank) averaged over questions with a hit
        ranks = [r["retrieval"]["rank"] for r in items if r["retrieval"]["rank"]]

        if ranks:
            mrr = sum(1/r for r in ranks) / n  # divide by n, not len(ranks), to penalise misses
            print(f"  MRR: {mrr:.3f}")
        else:
            print(f"  MRR: 0.000 (no hits)")

        #keyword coverage

        covs = [r["generation"]["keyword_coverage"] for r in items if r["generation"]["keyword_coverage"] is not None]
        if covs:
            avg_cov = sum(covs) / len(covs)
            print(f"  Keyword coverage: {avg_cov:.2%} avg")

    #Overall summary
    print("\n" + "=" * 60)
    print("OVERALL")
    print("=" * 60)
    in_scope = [r for r in results if r["category"] != "out_of_scope"]
    hits = sum(1 for r in in_scope if r["retrieval"]["hit"])
    ranks = [r["retrieval"]["rank"] for r in in_scope if r["retrieval"]["rank"]]
    mrr = sum(1/r for r in ranks) / len(in_scope) if in_scope else 0
    covs = [r["generation"]["keyword_coverage"] for r in in_scope if r["generation"]["keyword_coverage"] is not None]
    print(f"  Recall@{k}: {hits}/{len(in_scope)} ({100*hits/len(in_scope):.0f}%)")
    print(f"  MRR: {mrr:.3f}")
    print(f"  Keyword coverage: {sum(covs)/len(covs):.2%}" if covs else "  Keyword coverage: n/a")

    # Save versioned results
    from datetime import datetime
    runs_dir = Path("eval/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = runs_dir / f"run_{timestamp}.json"

    # Build a meta block describing the config that produced these results
    meta = {
        "timestamp": datetime.now().isoformat(),
        "k": k,
        "testset_path": testset_path,
        "n_questions": len(results),
        "model": "llama3.2:3b",
        "embedder": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 800,
        "chunk_overlap": 100,
        "summary": {
            "recall_at_k": round(hits / len(in_scope), 3) if in_scope else None,
            "mrr": round(mrr, 3),
            "keyword_coverage": round(sum(covs) / len(covs), 3) if covs else None,
        },
    }

    payload = {"meta": meta, "results": results}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nFull results saved to {out_path}")

if __name__ == "__main__":
    run_evaluation()
