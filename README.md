# 🤖 Türkçe RAG Asistan (MEB Ders Kitapları)

Bu proje, Türkçe ders kitapları üzerinden anlam tabanlı arama yaparak **kaynaklı cevap üreten bir RAG (Retrieval-Augmented Generation)** sistemidir.

📌 Özellikle MEB 9-10-11. sınıf ders kitapları gibi uzun ve yapılandırılmış içeriklerde çalışacak şekilde tasarlanmıştır.

---

# 🚀 Proje Amacı

* Türkçe NLP alanında güçlü bir sistem geliştirmek
* RAG mimarisini uçtan uca uygulamak
* Yerel (offline) çalışan bir AI asistan oluşturmak
* TEKNOFEST gibi yarışmalar için temel oluşturmak

---

# 🧠 Sistem Mimarisi

```text
PDF (MEB kitapları)
→ Temizleme & chunking
→ Embedding (E5)
→ Qdrant (Vector DB)
→ Retrieval (Top-K)
→ Reranker (BGE)
→ LLM (DeepSeek)
→ Cevap + Kaynak
```

---

# ⚙️ Kullanılan Teknolojiler

## 🔍 Embedding

* `intfloat/multilingual-e5-large`
* 1024 boyut
* Türkçe için güçlü semantic search

## 🧠 Reranker

* `BAAI/bge-reranker-v2-m3`
* Retrieval sonuçlarını yeniden sıralar
* Accuracy’yi ciddi artırır

## 💬 LLM

* `deepseek-r1:7b` (Ollama üzerinden)
* Türkçe cevap üretimi + reasoning

## 🗄️ Vector Database

* Qdrant

## 🎨 Arayüz

* Streamlit

---

# 📂 Proje Yapısı

```text
nlp-project/
│
├── app/
│   ├── ingest.py          # Veri işleme & embedding
│   ├── rag_pipeline.py    # RAG pipeline
│   └── ui.py              # Streamlit arayüz
│
├── data/
│   ├── raw/               # PDF dosyaları
│   └── clean/             # temizlenmiş metin
│
├── requirements.txt
└── README.md
```

---

# 🧹 Veri İşleme (Ingestion)

## Yapılan işlemler:

* PDF → text extraction
* Gürültü temizleme (OCR hataları, karekod vb.)
* Ünite ve konu tespiti
* Chunking (250 token + overlap)
* Enriched metadata ekleme:

```text
Sınıf
Ders
Ünite
Konu
Sayfa
```

* Embedding üretimi (E5 modeli)
* Qdrant’a kaydetme

---

# 🔍 RAG Pipeline

## 1. Retrieval

```text
Top 20 chunk alınır
```

## 2. Reranking

```text
En iyi 3 chunk seçilir
```

## 3. Prompt Engineering

* Soru tipine göre özel prompt:

  * Tanım
  * Açıklama
  * Karşılaştırma
  * Örnek

## 4. LLM

* DeepSeek ile cevap üretimi

## 5. Output

```text
Cevap + Kaynak
```

---

# ▶️ Kurulum

```bash
git clone https://github.com/farukince/nlp-example.git
cd nlp-example

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

# 🐳 Qdrant başlat

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

# 🧠 DeepSeek indir

```bash
ollama pull deepseek-r1:7b
```

---

# 📥 Veri yükleme

PDF dosyalarını buraya koy:

```text
data/raw/
```

---

# 🔄 Ingestion

```bash
python app/ingest.py
```

---

# 🧪 Test (terminal)

```bash
python app/rag_pipeline.py
```

---

# 🎨 UI başlat

```bash
streamlit run app/ui.py
```

---

# 🌐 Tarayıcı

```text
http://localhost:8501
```

---

# 📌 Örnek Sorular

```text
Arkhe nedir?
Sokrates'in ahlak anlayışı nedir?
Platon'a göre idealar nedir?
Aristoteles'in dört neden teorisi nedir?
Herakleitos ve Parmenides değişim konusunda nasıl ayrılır?
```

---

# ⚠️ Önemli Notlar

* Embedding değişirse → ingest tekrar çalıştırılmalı
* Qdrant reset gerekebilir
* PDF’ler GitHub’a yüklenmez

---

# 📈 Geliştirme Fikirleri

* ChatGPT tarzı konuşma arayüzü
* Çoklu PDF desteği (multi-subject)
* Evaluation metriği
* RAG + Agent sistemi
* API servisi (FastAPI)

---

# 🎯 Proje Seviyesi

```text
✔ RAG pipeline
✔ Semantic search
✔ Reranker
✔ LLM entegrasyonu
✔ UI
```

👉 Bu proje **orta-ileri seviye NLP / AI projesidir**

---


