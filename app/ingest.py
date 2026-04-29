from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import os
import re
import uuid

DATA_PATH = "data/raw"
CLEAN_PATH = "data/clean"
COLLECTION_NAME = "documents"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE = 384

model = SentenceTransformer(EMBEDDING_MODEL)
qdrant = QdrantClient("localhost", port=6333)


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("￾", "")
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(r"0:00\s*/\s*0:00", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([a-zçğıöşü])([A-ZÇĞİÖŞÜ])", r"\1 \2", text)
    # Çok kısa/gürültülü satırları azaltmak için temel temizlik
    noise_words = [
        "IMAGE FOR PAGE",
        "PARSED TEXT FOR PAGE",
    ]

    for word in noise_words:
        text = text.replace(word, "")

    return text.strip()


def read_pdf(file_path: str) -> list[dict]:
    reader = PdfReader(file_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)

        if len(cleaned) > 100:
            pages.append({
                "page": page_number,
                "text": cleaned
            })

    return pages


def save_clean_text(file_name: str, pages: list[dict]):
    os.makedirs(CLEAN_PATH, exist_ok=True)

    clean_file_name = file_name.replace(".pdf", "_clean.txt")
    clean_file_path = os.path.join(CLEAN_PATH, clean_file_name)

    with open(clean_file_path, "w", encoding="utf-8") as f:
        for page in pages:
            f.write(f"\n\n[PAGE {page['page']}]\n")
            f.write(page["text"])

    print(f"Clean text saved: {clean_file_path}")


def chunk_pages(pages: list[dict], chunk_size=260, overlap=60) -> list[dict]:
    chunks = []

    for page in pages:
        words = page["text"].split()
        start = 0
        chunk_index = 0

        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end]).strip()

            if len(chunk_text) > 200:
                chunks.append({
                    "text": chunk_text,
                    "page": page["page"],
                    "chunk_index": chunk_index
                })

            start += chunk_size - overlap
            chunk_index += 1

    return chunks


def recreate_collection():
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )


def ingest():
    recreate_collection()

    for file_name in os.listdir(DATA_PATH):
        if not file_name.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(DATA_PATH, file_name)
        print(f"Processing: {file_name}")

        pages = read_pdf(file_path)
        save_clean_text(file_name, pages)

        chunks = chunk_pages(pages)
        print(f"Chunks created: {len(chunks)}")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = model.encode(texts, show_progress_bar=True)

        points = []

        for chunk, vector in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload={
                        "text": chunk["text"],
                        "source": file_name,
                        "page": chunk["page"],
                        "chunk_index": chunk["chunk_index"],
                        "embedding_model": EMBEDDING_MODEL,
                    }
                )
            )

        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    print("Ingestion tamamlandı.")


if __name__ == "__main__":
    ingest()