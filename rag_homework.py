# rag_homework.py
import os
import sys
import argparse
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_chroma import Chroma

# Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# LLMs
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain.chains import RetrievalQA


BOOK_PATH = "book.txt"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_MODEL = "text-embedding-3-small"                  # OpenAI (jika ada kuota)
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fallback lokal
CHAT_MODEL = "gpt-4o-mini"                               # jika --llm openai

def die(msg: str, code: int = 1):
    print(f"[ERROR] {msg}")
    sys.exit(code)

parser = argparse.ArgumentParser()
parser.add_argument("--retrieval-only", action="store_true",
                    help="Hanya tampilkan hasil retrieve (tanpa LLM).")
parser.add_argument("--llm", choices=["openai", "ollama"], default="ollama",
                    help="Pilih backend LLM. Default: ollama (lokal, gratis).")
parser.add_argument("--ollama-model", default="mistral",
                    help="Nama model Ollama (mistral, llama3.1, qwen2.5:7b-instruct, dll).")
parser.add_argument("--ollama-base-url",
                    default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                    help="Base URL Ollama. Default dari env OLLAMA_BASE_URL atau http://127.0.0.1:11434")
args, _ = parser.parse_known_args()

def build_embeddings(docs: List[Document]):
    print("[5/7] Bangun vector store (Chroma) ...")
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY tidak ada")
        emb = OpenAIEmbeddings(model=EMBED_MODEL)
        vs = Chroma.from_documents(docs, emb)
        print("       Embeddings: OpenAI OK")
        return vs
    except Exception as e:
        err = str(e)
        if any(k in err.lower() for k in ["insufficient_quota", "429", "quota", "apikey", "openai"]):
            print("       [WARN] OpenAI embeddings gagal/kuota habis. Fallback ke HuggingFace (lokal).")
        else:
            print(f"       [INFO] Tidak pakai OpenAI embeddings ({e}). Pindah ke HuggingFace lokal.")
        emb = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
        vs = Chroma.from_documents(docs, emb)
        print(f"       Embeddings: HuggingFace OK ({HF_EMBED_MODEL})")
        return vs

def build_llm_and_chain(retriever):
    print("[7/7] Buat QA chain ...")
    if args.llm == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            die("OPENAI_API_KEY belum diset, atau gunakan --llm ollama.")
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
        print(f"       LLM: OpenAI ({CHAT_MODEL})")
    else:
        llm = ChatOllama(model=args.ollama_model, base_url=args.ollama_base_url, temperature=0)
        print(f"       LLM: Ollama ({args.ollama_model}) @ {args.ollama_base_url}")
    qa = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type="stuff")
    return qa

def main():
    print("[1/7] Cek environment ...")
    if args.llm == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("       [WARN] --llm openai dipilih tapi OPENAI_API_KEY tidak ada (kemungkinan gagal).")
    if args.llm == "ollama":
        print(f"       [INFO] OLLAMA_BASE_URL = {args.ollama_base_url}")

    print(f"[2/7] Cek file knowledge base: {BOOK_PATH} ...")
    if not os.path.exists(BOOK_PATH):
        die(f"{BOOK_PATH} tidak ditemukan. Taruh file book.txt di folder ini.")

    print("[3/7] Load teks ...")
    with open(BOOK_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        die("book.txt kosong.")

    print(f"[4/7] Split jadi chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)
    if not chunks:
        die("Gagal split teks.")
    docs = [Document(page_content=c) for c in chunks]
    print(f"       Total chunks: {len(docs)}")

    vs = build_embeddings(docs)

    print("[6/7] Buat retriever ...")
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    if args.retrieval_only:
        print("[7/7] Mode retrieval-only. Ketik pertanyaan (exit/quit untuk keluar).")
        while True:
            try:
                q = input("Q: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nKeluar."); break
            if q.lower() in {"exit", "quit"}:
                print("Keluar."); break
            if not q:
                continue
            try:
                results: List[Document] = retriever.invoke(q)  # retriever adalah Runnable
                print("Top-4 potongan konteks:")
                for i, d in enumerate(results, 1):
                    snippet = d.page_content[:300].replace("\n", " ")
                    print(f"[{i}] {snippet}{'...' if len(d.page_content)>300 else ''}")
                print("(LLM dimatikan; jalankan tanpa --retrieval-only untuk jawaban.)")
            except Exception as e:
                print(f"[ERR retrieve] {e}")
        return

    qa = build_llm_and_chain(retriever)

    print("Ready. Ketik pertanyaan (exit/quit untuk keluar).")
    while True:
        try:
            q = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKeluar."); break
        if q.lower() in {"exit", "quit"}:
            print("Keluar."); break
        if not q:
            continue
        try:
            out = qa.invoke({"query": q})   # API baru LangChain
            print("A:", out.get("result", out))
        except Exception as e:
            print(f"[ERR jawab] {e}")

if __name__ == "__main__":
    main()
