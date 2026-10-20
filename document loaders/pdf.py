from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

model=GoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.3
)
data=PyPDFLoader("GRU.pdf")

docs=data.load()
template=ChatPromptTemplate.from_messages(
    [
        ("system","you are the AI that summarizes the text"),
        ("human","{data}")
    ]
)

prompt=template.format_messages(data=docs)
result=model.invoke(prompt)

print(result)
