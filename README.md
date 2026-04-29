# Türkçe RAG Asistan 🤖

Bu proje, Türkçe dokümanlar üzerinden anlam tabanlı arama yaparak kaynaklı cevap üreten bir RAG (Retrieval-Augmented Generation) sistemidir.

## 🚀 Özellikler

* PDF dokümanlardan veri çıkarma
* Metin temizleme ve chunking
* Embedding ile anlam tabanlı arama
* Qdrant vector database kullanımı
* Ollama ile yerel LLM üzerinden cevap üretimi
* Streamlit ile kullanıcı arayüzü

## 🧠 Kullanılan Teknolojiler

* Python
* Sentence Transformers
* Qdrant
* Ollama (Qwen2.5)
* Streamlit

## ⚙️ Kurulum

```bash
git clone https://github.com/kullaniciadi/nlp-example.git
cd nlp-example

python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## ▶️ Çalıştırma

```bash
streamlit run app/ui.py
```

## 📌 Örnek Soru

```text
Osmanlı İmparatorluğu ne zaman kuruldu?
```

## 🎯 Amaç

Türkçe NLP ve RAG sistemleri üzerine deneysel bir proje geliştirmek ve TEKNOFEST yarışmasına hazırlık yapmak.
