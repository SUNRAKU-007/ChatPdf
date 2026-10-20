# Chat with your PDF

A RAG app: upload a PDF and ask questions about it. Built with Streamlit, LangChain, ChromaDB, sentence-transformers embeddings and Gemini.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then put your real GEMINI_API_KEY in .env
streamlit run app.py
```
