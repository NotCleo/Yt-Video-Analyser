import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:3.8b")

embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = Ollama(
        model=OLLAMA_MODEL,
        temperature=0.1,
    )

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that can answer questions about YouTube videos
        based on the video's transcript.

        Answer the following question: {question}
        By searching the following video transcript: {docs}

        Only use factual information from the transcript to answer the question.

        If you feel you don't have enough information to answer the question, say "I don't know."

        Your answers should be verbose and detailed.
        """,
    )

    chain = prompt | llm

    response = chain.invoke({
        "question": query,
        "docs": docs_page_content
    })

    response = response.replace("\n", "")
    return response, docs
