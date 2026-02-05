import streamlit as st
import chromadb
import plotly.express as px
from chromadb.utils import embedding_functions as ef
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import COLLECTION, LOCAL_LLM, TOP_K, EMBED_MODEL, CHROMA_DIR, SYSTEM_PROMPT, APP_TITLE


# this is base folder
@st.cache_resource(show_spinner=False)
def load_local_model():
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_LLM,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model


tokenizer, model = load_local_model()

@st.cache_resource(show_spinner=False)
def get_collection():
    embedding_fn = ef.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION, embedding_function=embedding_fn)


def local_llm(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

if "history" not in st.session_state:
    st.session_state.history = []

st.write("Model loaded:", "tokenizer" in globals())

question = st.text_input("Ask a question about the documents:")

collection = get_collection()

if st.button("Ask") and question:
    st.write("Checkpoint A: Button pressed")

    # Retrieve relevant chunks
    res = collection.query(query_texts=[question], n_results=TOP_K)
    st.write("Checkpoint B: Chroma query done")

    chunks = list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))
    st.write("Checkpoint C: Chunks zipped")

    # Build context
    context = "\n\n---\n\n".join(c[0] for c in chunks)
    prompt = (
        f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\n"
        f"User question: {question}\nAnswer:"
    )
    st.write("Checkpoint D: Prompt built")
    reply = local_llm(prompt)
    st.write("Checkpoint E: LLM returned")



# this shows the history of questions and answers + sources
for idx, item in enumerate(reversed(st.session_state.history)):
    # extra credit: creating distance vizualization
    # added button to show/hide
    if st.button("Show Distances Visualization", key=f"dist_viz_{idx}"):
        texts = [chunk[0] for chunk in item["chunks"]]
        metadatas = [chunk[1] for chunk in item["chunks"]]
        distances = [chunk[2] for chunk in item["chunks"]]

        data = []
        for i, (text, meta, dist) in enumerate(zip(texts, metadatas, distances)):
            data.append({
                "Index": i,
                "Distance": dist,
                "Preview": text[:200].replace("\n", " ") + "…",
                "Source": meta.get("source", "unknown")
            })
        fig = px.scatter(
            data,
            x="Index",
            y="Distance",
            color="Source",
            hover_data={"Preview": True, "Distance": True, "Index": False},
            title="Similarity Distance of Retrieved Chunks"
        )
        fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='SteelBlue')))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Q:** {item['q']}")
    st.markdown(f"**A:** {item['a']}")
    with st.expander("Sources"):
        for txt, meta, dist in item["chunks"]:
            st.write(f"- *{meta['source']}* (dist={dist:.3f})\n> {txt[:300]}…")