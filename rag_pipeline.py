import os
import logging
from typing import List, Optional, Any, Mapping

logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.api").setLevel(logging.ERROR)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM

from groq import Groq


# ---------- Custom Groq LLM ----------
class GroqLLM(LLM):
    model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.7
    max_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "groq"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in your environment.")

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        text = response.choices[0].message.content

        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
                    break

        return text


# ---------- Create RAG chain ----------
def create_chain():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="hf_cache",
        encode_kwargs={"normalize_embeddings": True},
    )

    if not os.path.exists("db"):
        loader = TextLoader("speech.txt")
        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory="db")
        vectordb.persist()
    else:
        vectordb = Chroma(persist_directory="db", embedding_function=embedding_model)

    retriever = vectordb.as_retriever(search_kwargs={"k": 1})

    llm = GroqLLM()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
    )
    return qa_chain
