from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

from rag.hybrid_retriever import hybrid_search
from rag.reranker import rerank


# 🔹 Load model
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_length=120,
    truncation=True
)

llm = HuggingFacePipeline(pipeline=pipe)


# 🔹 Clean PDF noise
def clean_text(text: str) -> str:
    lines = text.split("\n")

    bad_words = [
        "reprint", "unit", "indd", "grade",
        "page", "exercise", "chapter", "let us"
    ]

    clean_lines = []

    for line in lines:
        line = line.strip()

        if len(line) < 25:
            continue

        if any(word in line.lower() for word in bad_words):
            continue

        clean_lines.append(line)

    return " ".join(clean_lines)


# 🔹 Keyword overlap scoring (improves retrieval)
def keyword_score(query: str, text: str) -> int:
    q = set(query.lower().split())
    t = set(text.lower().split())
    return len(q & t)


# 🔹 Build context from best chunk
def build_context(docs):
    if not docs:
        return ""

    best_doc = docs[0]
    return clean_text(best_doc.page_content)


# 🔥 MAIN FUNCTION
def generate_answer(query, subject=None):

    # 1️⃣ Retrieve documents
    docs = hybrid_search(query, k=8)

    # 2️⃣ Subject filtering
    if subject:
        docs = [
            d for d in docs
            if d.metadata.get("subject", "").lower() == subject.lower()
        ] or docs

    # 3️⃣ Keyword boost (VERY IMPORTANT)
    docs = sorted(
        docs,
        key=lambda d: keyword_score(query, d.page_content),
        reverse=True
    )

    # 4️⃣ Rerank (select best 1 chunk)
    top_docs = rerank(query, docs, top_k=1)

    if not top_docs:
        return {
            "answer": "Not found in provided material.",
            "subject": "Unknown",
            "source": "Unknown"
        }

    best_doc = top_docs[0]

    context = clean_text(best_doc.page_content)

    subject_name = best_doc.metadata.get("subject", "Unknown")
    source = best_doc.metadata.get("source", "Unknown")

    # 🔥 STRONG PROMPT (NO LEAK)
    prompt = f"""
You are a Grade 6 tutor.

Read the context and answer the question.

RULES:
- Use only the context
- Do NOT repeat instructions
- Do NOT copy the question
- Give only the final answer
- If answer not found → say: Not found in provided material

Context:
{context}

Question: {query}

Final Answer:
"""

    # 5️⃣ Generate
    answer = llm.invoke(prompt).strip()

    # 🔥 CLEAN OUTPUT (IMPORTANT)
    if answer.lower().startswith("answer"):
        answer = answer.replace("Answer:", "").strip()

    if "answer in simple" in answer.lower():
        answer = "Not found in provided material."

    if len(answer) < 5:
        answer = "Not found in provided material."

    return {
        "answer": answer,
        "subject": subject_name,
        "source": source
    }