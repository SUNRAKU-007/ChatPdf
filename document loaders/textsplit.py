from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import CharacterTextSplitter

splitter=CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1
)

load_dotenv()

model=GoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.3
)
data=TextLoader("notes1.txt")

docs=data.load()

chunks=splitter.split_documents(docs)

print(len(chunks))

for i in chunks:
    print(i.page_content)
    print()
    print()
