import json
import math
import os
from typing import List, Sequence, Tuple

import streamlit as st
import torch

st.set_page_config(
    page_title="A6 RAG Assistant",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main > div {
        max-width: 940px;
        padding-top: 1.4rem;
    }
    .hero {
        border: none;
        background: transparent;
        border-radius: 0;
        padding: 4px 0 10px 0;
        margin-bottom: 10px;
    }
    .hero h1 {
        font-size: 1.45rem;
        margin: 0;
        color: #e5edf8;
    }
    .hero p {
        margin: 6px 0 0 0;
        color: #b8c4d8;
        font-size: 0.95rem;
    }
    .intro-hint {
        background: rgba(37, 99, 235, 0.22);
        border: 1px solid rgba(96, 165, 250, 0.5);
        color: #dbeafe;
        border-radius: 12px;
        padding: 12px 14px;
        margin: 8px 0 4px 0;
        font-size: 1.05rem;
        line-height: 1.45;
    }
    .mode-chip {
        display: inline-block;
        border: 1px solid #cfd8e3;
        background: #f4f7fb;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.78rem;
        color: #344054;
        margin-top: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)


DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
CHAPTER_TXT_PATH = os.path.join(DATASET_DIR, "chapter11.txt")
ENRICHED_CHUNKS_PATH = os.path.join(DATASET_DIR, "enriched_chunks.json")


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    embed_mod = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(device)
    embed_mod.eval()

    gen_tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token

    gen_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    gen_mod = AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct",
        dtype=gen_dtype,
        low_cpu_mem_usage=False,
    )
    gen_mod.to(device)
    gen_mod.eval()
    gen_mod.generation_config.pad_token_id = gen_tok.pad_token_id
    gen_mod.generation_config.eos_token_id = gen_tok.eos_token_id

    return device, embed_tok, embed_mod, gen_tok, gen_mod


def get_embedding(text: str) -> List[float]:
    device, embed_tok, embed_mod, _, _ = load_models()
    inputs = embed_tok(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        out = embed_mod(**inputs)
    return out.last_hidden_state[:, 0, :].squeeze().cpu().tolist()


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-10)


@st.cache_resource(show_spinner="Building Naive RAG vector database...")
def load_naive_vector_db() -> List[Tuple[str, List[float]]]:
    if not os.path.exists(CHAPTER_TXT_PATH):
        return []
    with open(CHAPTER_TXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size=300, overlap=50)
    return [(chunk, get_embedding(chunk)) for chunk in chunks]


@st.cache_resource(show_spinner="Building Contextual Retrieval vector database...")
def load_contextual_vector_db() -> List[Tuple[str, List[float]]]:
    if not os.path.exists(ENRICHED_CHUNKS_PATH):
        return []
    with open(ENRICHED_CHUNKS_PATH, "r", encoding="utf-8") as f:
        enriched_chunks = json.load(f)

    return [(chunk, get_embedding(chunk)) for chunk in enriched_chunks]


def retrieve(query: str, vector_db: List[Tuple[str, List[float]]], top_n: int) -> List[Tuple[str, float]]:
    q_emb = get_embedding(query)
    scored = [(chunk, cosine_similarity(q_emb, emb)) for chunk, emb in vector_db]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def generate_answer(query: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
    device, _, _, gen_tok, gen_mod = load_models()
    context = "\n".join(f"- {chunk}" for chunk, _ in retrieved_chunks)
    prompt = (
        "You are an academic assistant for machine translation (Chapter 11).\n"
        "Answer using only the provided context. If the context is insufficient, say so clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    inputs = gen_tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_len = inputs["input_ids"].shape[-1]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = gen_mod.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=220,
            do_sample=False,
            pad_token_id=gen_tok.pad_token_id,
            eos_token_id=gen_tok.eos_token_id,
        )
    answer = gen_tok.decode(out_ids[0][input_len:], skip_special_tokens=True).strip()
    return answer if answer else "Unable to generate an answer from the retrieved context."


st.markdown(
    """
<div class="hero">
  <h1>Machine Translation Retrieval Assistant</h1>
  <p>Assignment 6 web application for Chapter 11 Q&A using Naive RAG and Contextual Retrieval.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Configuration")
    retrieval_mode = st.radio(
        "Retrieval method",
        options=["Naive RAG", "Contextual Retrieval"],
        help="Switch between baseline chunk retrieval and context-enriched retrieval.",
    )
    top_n = st.slider("Top-N chunks", min_value=1, max_value=10, value=5)
    st.divider()
    st.caption("Retriever: BAAI/bge-small-en-v1.5")
    st.caption("Generator: unsloth/Llama-3.2-1B-Instruct")
    st.caption("Context Enrichment: Gemini 2.0 Flash (precomputed)")


if retrieval_mode == "Naive RAG":
    vector_db = load_naive_vector_db()
    missing_msg = (
        "Naive RAG data is not available. Execute the notebook to generate datasets/chapter11.txt."
    )
else:
    vector_db = load_contextual_vector_db()
    missing_msg = (
        "Contextual Retrieval data is not available. Execute the notebook to generate datasets/enriched_chunks.json."
    )

if not vector_db:
    st.error(missing_msg)
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            st.markdown(
                f"<span class='mode-chip'>{msg.get('mode', 'Unknown mode')}</span>",
                unsafe_allow_html=True,
            )
            sources = msg.get("sources", [])
            if sources:
                with st.expander("Source Chunks", expanded=False):
                    for idx, (chunk, score) in enumerate(sources, start=1):
                        st.markdown(f"**Chunk {idx}** | Similarity: `{score:.4f}`")
                        st.caption(chunk[:700] + ("..." if len(chunk) > 700 else ""))

prompt = st.chat_input("Ask a question about Chapter 11 (Machine Translation)")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and generating answer..."):
            retrieved = retrieve(prompt, vector_db, top_n=top_n)
            answer = generate_answer(prompt, retrieved)
        st.markdown(answer)
        st.markdown(
            f"<span class='mode-chip'>{retrieval_mode}</span>",
            unsafe_allow_html=True,
        )
        with st.expander("Source Chunks", expanded=False):
            for idx, (chunk, score) in enumerate(retrieved, start=1):
                st.markdown(f"**Chunk {idx}** | Similarity: `{score:.4f}`")
                st.caption(chunk[:700] + ("..." if len(chunk) > 700 else ""))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "mode": retrieval_mode,
            "sources": retrieved,
        }
    )
elif not st.session_state.messages:
    st.markdown(
        """
<div class="intro-hint">
Ask a chapter-specific question such as: 'What problem does attention solve in encoder-decoder MT?'
</div>
""",
        unsafe_allow_html=True,
    )
