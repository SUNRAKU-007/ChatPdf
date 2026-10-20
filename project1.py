import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Hugging Face Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 2. Load Existing Chroma Vector Database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# 3. MMR Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# 4. Gemini LLM
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key
)

# 5. Prompt Template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}
"""
    )
])

print("RAG system created!")
print("Press 0 to exit.")

# 6. Chat Loop
while True:
    query = input("\nYou: ").strip()

    if query == "0":
        break

    if not query:
        continue

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    if not docs:
        print("\nAI: I could not find the answer in the document.")
        continue

    # Display retrieved documents
    print("\n--- Retrieved Documents ---")

    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print(doc.page_content[:300])

    # Construct context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Generate answer
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}")

