import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


DATA_DIR = "data"


def load_documents():
    documents = []

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    for subject in os.listdir(DATA_DIR):
        subject_path = os.path.join(DATA_DIR, subject)

        if not os.path.isdir(subject_path):
            continue

        for file_name in os.listdir(subject_path):
            if not file_name.lower().endswith(".pdf"):
                continue

            file_path = os.path.join(subject_path, file_name)
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["subject"] = subject.strip()
                doc.metadata["source"] = file_name
                doc.metadata["file_path"] = file_path

            documents.extend(docs)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} pages")
    print(f"Created {len(chunks)} chunks")