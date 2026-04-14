from sentence_transformers import CrossEncoder

# Load once (global)
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, docs, top_k=2):
    """
    Rerank retrieved documents using CrossEncoder
    """

    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]

    scores = reranker_model.predict(pairs)

    # Attach scores
    scored_docs = list(zip(docs, scores))

    # Sort by score
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Return top_k docs
    return [doc for doc, _ in scored_docs[:top_k]]