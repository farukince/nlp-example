from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama

def generate_answer(question, results):
    context = ""

    sources = []

    for r in results[:2]:
        text = r.payload.get("text", "")
        page = r.payload.get("page", "")
        source = r.payload.get("source", "")

        context += text + "\n\n"
        sources.append(f"{source} - Sayfa {page}")

    prompt = f"""
Sen kaynaklara sıkı sıkıya bağlı Türkçe bir RAG asistanısın.

Kurallar:
- Sadece verilen Bilgiler bölümündeki ifadeleri kullan.
- Bilgilerde açıkça yazmayan şeyi ekleme.
- Farklı tarih veya görüş varsa hepsini ayrı ayrı belirt.
- Tarihleri karıştırma.
- Cevap kısa, net ve maddeli olsun.
- Emin olmadığın noktada "Kaynakta bu şekilde açıkça belirtilmemiştir." de.

Bilgiler:
{context}

Soru:
{question}

Cevap formatı:
- Kısa cevap:
- Ayrıntı:
- Kaynak:
"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    sources_text = "\n".join(set(sources))

    return f"{answer}\n\nKaynak:\n{sources_text}"

COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(EMBEDDING_MODEL)
qdrant = QdrantClient("localhost", port=6333)


def search_documents(question: str, top_k: int = 5):
    question_vector = model.encode(question).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=question_vector,
        limit=top_k
    ).points

    return results


def print_results(results):
    print("\nEn alakalı kaynak parçaları:\n")

    for i, result in enumerate(results, start=1):
        payload = result.payload

        print("=" * 80)
        print(f"Sonuç {i}")
        print(f"Skor: {result.score:.4f}")
        print(f"Kaynak: {payload.get('source')}")
        print(f"Sayfa: {payload.get('page')}")
        print(f"Chunk index: {payload.get('chunk_index')}")
        print("-" * 80)
        print(payload.get("text", "")[:1200])
        print()


if __name__ == "__main__":
    question = input("Sorunu yaz: ")

    results = search_documents(question)
    print_results(results)

    print("\n" + "="*80)
    print("AI CEVAP:\n")

    answer = generate_answer(question, results)
    print(answer)