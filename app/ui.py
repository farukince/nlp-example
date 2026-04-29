import streamlit as st
from rag_pipeline import search_documents, generate_answer

st.set_page_config(
    page_title="Türkçe RAG Asistan",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Türkçe RAG Asistan")
st.caption("PDF dokümanları üzerinden kaynaklı cevap üreten mini RAG demo.")

question = st.text_input(
    "Sorunu yaz:",
    placeholder="Örn: Osmanlı İmparatorluğu ne zaman kuruldu?"
)

top_k = st.slider("Kaç kaynak parçası aransın?", 1, 10, 5)

if st.button("Cevapla", type="primary"):
    if not question.strip():
        st.warning("Lütfen bir soru yaz.")
    else:
        with st.spinner("Kaynaklar aranıyor ve cevap üretiliyor..."):
            results = search_documents(question, top_k=top_k)
            answer = generate_answer(question, results)

        st.subheader("AI Cevap")
        st.write(answer)

        st.divider()

        st.subheader("Kaynak")

        sources = set()

        for r in results[:2]:
            source = r.payload.get("source")
            page = r.payload.get("page")
            sources.add(f"{source} - Sayfa {page}")

        for s in sources:
            st.write(f"• {s}")