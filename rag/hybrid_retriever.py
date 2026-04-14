import os
import pickle

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from rag.ingest import load_documents, split_documents


VECTORSTORE_DIR = "vectorstore"
BM25_FILE = "bm25.pkl"

bm25 = None
bm25_docs = None
vectorstore = None


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def detect_subject(query: str):
    q = query.lower()

    english_keywords = [
        "meaning", "word", "poem", "story", "author", "synonym",
        "antonym", "english", "sentence", "grammar", "strayed"
    ]
    hindi_keywords = [
        "hindi", "हिंदी", "शब्द", "कविता", "पाठ", "अर्थ"
    ]
    mathematics_keywords = [
        "math", "mathematics", "addition", "subtraction", "multiply",
        "division", "fraction", "decimal", "angle", "perimeter",
        "area", "number", "equation"
    ]
    science_keywords = [
        "science", "plant", "animal", "photosynthesis", "water", "air",
        "food", "habitat", "body", "motion", "electricity", "light", "force"
    ]
    social_science_keywords = [
        "history", "geography", "civics", "continent", "continents",
        "ocean", "oceans", "map", "earth", "king", "empire",
        "government", "social science", "globe", "climate"
    ]

    if any(word in q for word in english_keywords):
        return "english"
    if any(word in q for word in hindi_keywords):
        return "hindi"
    if any(word in q for word in mathematics_keywords):
        return "mathematics"
    if any(word in q for word in science_keywords):
        return "science"
    if any(word in q for word in social_science_keywords):
        return "social science"

    return None


def build_and_save():
    global bm25, bm25_docs, vectorstore

    print("🔄 Loading documents...")
    docs = load_documents()
    print(f"📄 Loaded {len(docs)} pages")

    print("✂️ Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"🧩 Created {len(chunks)} chunks")

    print("📊 Building BM25 index...")
    tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_docs = chunks

    with open(BM25_FILE, "wb") as f:
        pickle.dump((bm25, bm25_docs), f)

    print("🧠 Building FAISS vectorstore...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)

    print("✅ Vectorstore and BM25 saved successfully")


def load_all():
    global bm25, bm25_docs, vectorstore

    if bm25 is None or bm25_docs is None:
        print("📂 Loading BM25...")
        with open(BM25_FILE, "rb") as f:
            bm25, bm25_docs = pickle.load(f)

    if vectorstore is None:
        print("📂 Loading FAISS vectorstore...")
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )


def hybrid_search(query: str, k: int = 5):
    load_all()

    print(f"🔍 Searching for: {query}")

    inferred_subject = detect_subject(query)

    # FAISS retrieval only for final output
    docs = vectorstore.similarity_search(query, k=10)

    # Subject filtering
    if inferred_subject:
        filtered_docs = [
            doc for doc in docs
            if doc.metadata.get("subject", "").strip().lower() == inferred_subject.lower()
        ]

        if filtered_docs:
            docs = filtered_docs

    print(f"📌 Retrieved {len(docs[:k])} documents")
    return docs[:k]


if __name__ == "__main__":
    if not os.path.exists(VECTORSTORE_DIR) or not os.path.exists(BM25_FILE):
        print("🚀 First-time setup: building database...")
        build_and_save()
    else:
        print("✅ Vectorstore already exists. Skipping build.")