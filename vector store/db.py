from langchain_community.vectorstores import Chroma

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()


from langchain_core.documents import Document

docs=[
    Document(page_content="Python is widely used in Artificial Intelligence.",metadata={"source":"AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.",metadata={"source":"DataScience_book"}),
    Document(page_content="Neural Networks are used in deep learning.",metadata={"source":"DL_book"})
]

embedding_model=GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


vectorstore=Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

result=vectorstore.similarity_search("what is used for data analysis?",k=2) # 'k' means how much results we want it to answer

# vectorstore is not responsible for answering our questions
#it is used for retrieving your info, LLM is responsible for answering

for r in result:
    print(r.page_content)
    print(r.metadata)
    
retriever=vectorstore.as_retriever()

docs=retriever.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)


