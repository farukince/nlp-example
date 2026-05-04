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

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
VECTOR_SIZE = 1024

CLASS_LEVEL = "11. Sınıf"
SUBJECT = "Felsefe"

CHUNK_SIZE = 250
CHUNK_OVERLAP = 60

model = SentenceTransformer(EMBEDDING_MODEL)
qdrant = QdrantClient("localhost", port=6333)


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("￾", "")
    text = text.replace("", "ı")
    text = text.replace("", "ı")
    text = text.replace("", "ğ")

    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"0:00\s*/\s*0:00", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(r"([a-zçğıöşü])([A-ZÇĞİÖŞÜ])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)

    noise_patterns = [
        "IMAGE FOR PAGE",
        "PARSED TEXT FOR PAGE",
        "Karekod",
        "akıllı cihazınıza",
        "Daha fazla içerik",
        "Ünite sunumu için",
    ]

    for noise in noise_patterns:
        text = text.replace(noise, "")

    return text.strip()


def detect_unit_title(text: str) -> str:
    patterns = [
        r"\d+\.\s*ÜNİTE[:\s]+[^\.]{5,120}?FELSEFESİ",
        r"MÖ\s*\d+.*?FELSEFESİ",
        r"MS\s*\d+.*?FELSEFESİ",
        r"\d+\.\s*ÜNİTE",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    return ""


def detect_topic_title(text: str) -> str:
    patterns = [
        r"\d+\.\d+\.\s+[A-ZÇĞİÖŞÜ0-9\s\-.,]+",
        r"(Sofistler ile Sokrates’in Bilgi ve Ahlak Anlayışları)",
        r"(Platon ve Aristoteles’in Varlık, Bilgi ve Değer Anlayışları)",
        r"(İlk Neden \(Arkhe\) ve Değişim Problemleri)",
        r"(Hristiyan Felsefesinin Temel Özellikleri ve Problemleri)",
        r"(İslam Felsefesinin Temel Özellikleri ve Problemleri)",
        r"(Evrenin Yaratılışı Problemi)",
        r"(Kötülük Problemi)",
        r"(Tümeller Problemi)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    return ""


def read_pdf(file_path: str) -> list[dict]:
    reader = PdfReader(file_path)
    pages = []

    current_unit = ""
    current_topic = ""

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)

        found_unit = detect_unit_title(cleaned)
        if found_unit:
            current_unit = found_unit

        found_topic = detect_topic_title(cleaned)
        if found_topic:
            current_topic = found_topic

        # Kapak, İstiklal Marşı, Gençliğe Hitabe, içindekiler vb.
        if page_number < 10:
            continue

        if len(cleaned) > 150:
            pages.append({
                "page": page_number,
                "text": cleaned,
                "unit": current_unit,
                "topic": current_topic
            })

    return pages


def save_clean_text(file_name: str, pages: list[dict]):
    os.makedirs(CLEAN_PATH, exist_ok=True)

    clean_file_name = file_name.replace(".pdf", "_clean.txt")
    clean_file_path = os.path.join(CLEAN_PATH, clean_file_name)

    with open(clean_file_path, "w", encoding="utf-8") as f:
        for page in pages:
            f.write(f"\n\n[PAGE {page['page']}]\n")
            if page.get("unit"):
                f.write(f"[UNIT] {page['unit']}\n")
            if page.get("topic"):
                f.write(f"[TOPIC] {page['topic']}\n")
            f.write(page["text"])

    print(f"Clean text saved: {clean_file_path}")


def chunk_pages(pages: list[dict], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[dict]:
    chunks = []

    for page in pages:
        words = page["text"].split()
        start = 0
        chunk_index = 0

        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end]).strip()

            if len(chunk_text) > 250:
                enriched_text = (
                    f"Ders: {SUBJECT}\n"
                    f"Sınıf: {CLASS_LEVEL}\n"
                    f"Ünite: {page.get('unit', '')}\n"
                    f"Konu: {page.get('topic', '')}\n"
                    f"Sayfa: {page['page']}\n"
                    f"Metin: {chunk_text}"
                )

                chunks.append({
                    "text": chunk_text,
                    "enriched_text": enriched_text,
                    "page": page["page"],
                    "unit": page.get("unit", ""),
                    "topic": page.get("topic", ""),
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

        if not chunks:
            print(f"No chunks found for {file_name}")
            continue

        embedding_texts = [
            f"passage: {chunk['enriched_text']}"
            for chunk in chunks
        ]

        embeddings = model.encode(
            embedding_texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        points = []

        for chunk, vector in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload={
                        "text": chunk["text"],
                        "enriched_text": chunk["enriched_text"],
                        "source": file_name,
                        "class_level": CLASS_LEVEL,
                        "subject": SUBJECT,
                        "unit": chunk["unit"],
                        "topic": chunk["topic"],
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