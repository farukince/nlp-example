from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama
import re

COLLECTION_NAME = "documents"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "deepseek-r1:7b"

RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 3

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)

qdrant = QdrantClient("localhost", port=6333)


def detect_question_type(question: str) -> str:
    q = question.lower()

    if any(x in q for x in ["karşılaştır", "fark", "benzerlik", "ayrım"]):
        return "compare"

    if any(x in q for x in ["örnek", "örnek ver"]):
        return "example"

    if any(x in q for x in ["nedir", "ne demek", "tanım"]):
        return "definition"

    if any(x in q for x in ["açıkla", "anlat", "nasıl", "neden"]):
        return "explain"

    return "general"


def search_documents(question: str, top_k: int = RETRIEVAL_TOP_K):
    query_text = f"query: {question}"

    question_vector = embedding_model.encode(
        query_text,
        normalize_embeddings=True
    ).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=question_vector,
        limit=top_k
    ).points

    return results


def rerank_documents(question: str, results, top_k: int = RERANK_TOP_K):
    if not results:
        return []

    pairs = []

    for result in results:
        payload = result.payload
        text = payload.get("enriched_text") or payload.get("text", "")
        pairs.append([question, text])

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True
    )

    final_results = []

    for result, rerank_score in ranked[:top_k]:
        result.payload["rerank_score"] = float(rerank_score)
        final_results.append(result)

    return final_results


def remove_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def build_prompt(question: str, context: str, question_type: str) -> str:
    base_rules = """
Sen MEB ders kitabına dayalı Türkçe eğitim asistanısın.

Genel kurallar:
- Sadece verilen Bilgiler bölümüne göre cevap ver.
- Kaynakta olmayan bilgiyi ekleme.
- Öğrenci seviyesine uygun, açık ve anlaşılır anlat.
- Desteklenmeyen bilgiyi çıkar.
- Gereksiz uzatma.
- <think> etiketi veya düşünme süreci yazma.
- Kaynakta bilgi yoksa "Bu bilgi verilen kaynakta açıkça bulunamadı." de.
"""

    if question_type == "definition":
        task_rules = """
Soru tipi: TANIM
Cevap formatı:
- Tanım:
- Kısa açıklama:
"""
    elif question_type == "compare":
        task_rules = """
Soru tipi: KARŞILAŞTIRMA
Cevap formatı:
- Benzerlikler:
- Farklılıklar:
- Sonuç:
"""
    elif question_type == "example":
        task_rules = """
Soru tipi: ÖRNEK
Cevap formatı:
- Cevap:
- Kaynaktaki örnekler:
- Açıklama:
Not: Kaynakta örnek yoksa örnek uydurma.
"""
    elif question_type == "explain":
        task_rules = """
Soru tipi: AÇIKLAMA
Cevap formatı:
- Cevap:
- Adım adım açıklama:
- Özet:
"""
    else:
        task_rules = """
Soru tipi: GENEL
Cevap formatı:
- Cevap:
- Kısa açıklama:
"""

    return f"""
{base_rules}

{task_rules}

Bilgiler:
{context}

Soru:
{question}
"""


def generate_answer(question: str, results):
    context = ""
    sources = []

    selected_results = results[:RERANK_TOP_K]

    for r in selected_results:
        payload = r.payload

        text = payload.get("text", "")
        source = payload.get("source", "")
        page = payload.get("page", "")
        unit = payload.get("unit", "")
        topic = payload.get("topic", "")
        class_level = payload.get("class_level", "")
        subject = payload.get("subject", "")

        context += (
            f"[Kaynak: {source}, {class_level} {subject}, Sayfa: {page}, Ünite: {unit}, Konu: {topic}]\n"
            f"{text}\n\n"
        )

        sources.append(
            f"{class_level} {subject} - {source} - Sayfa {page}"
        )

    question_type = detect_question_type(question)
    prompt = build_prompt(question, context, question_type)

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]
    answer = remove_think_blocks(answer)

    sources_text = "\n".join(sorted(set(sources)))

    return f"{answer}\n\nKaynak:\n{sources_text}"


def answer_question(question: str):
    retrieved_results = search_documents(question, top_k=RETRIEVAL_TOP_K)
    reranked_results = rerank_documents(question, retrieved_results, top_k=RERANK_TOP_K)
    answer = generate_answer(question, reranked_results)

    return answer, reranked_results


def print_results(results):
    print("\nEn alakalı kaynak parçaları:\n")

    for i, result in enumerate(results, start=1):
        payload = result.payload

        print("=" * 80)
        print(f"Sonuç {i}")
        print(f"Vector Skor: {result.score:.4f}")
        print(f"Rerank Skor: {payload.get('rerank_score')}")
        print(f"Kaynak: {payload.get('source')}")
        print(f"Sınıf: {payload.get('class_level')}")
        print(f"Ders: {payload.get('subject')}")
        print(f"Ünite: {payload.get('unit')}")
        print(f"Konu: {payload.get('topic')}")
        print(f"Sayfa: {payload.get('page')}")
        print(f"Chunk index: {payload.get('chunk_index')}")
        print("-" * 80)
        print(payload.get("text", "")[:1200])
        print()


if __name__ == "__main__":
    question = input("Sorunu yaz: ")

    answer, results = answer_question(question)

    print_results(results)

    print("\n" + "=" * 80)
    print("AI CEVAP:\n")
    print(answer)