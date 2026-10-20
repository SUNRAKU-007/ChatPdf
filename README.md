# Chat with your PDF

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatpdf-zxenoqf5ybmprnz42omsp6.streamlit.app/)

**Live demo:** https://chatpdf-zxenoqf5ybmprnz42omsp6.streamlit.app/


A RAG app: upload a PDF and ask questions about it. Built with Streamlit, LangChain, ChromaDB, sentence-transformers embeddings and Gemini.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then put your real GEMINI_API_KEY in .env
streamlit run app.py
```

