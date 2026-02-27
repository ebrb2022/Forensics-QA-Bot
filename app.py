import streamlit as st
import chromadb
import plotly.express as px
from chromadb.utils import embedding_functions as ef
from config import COLLECTION, LOCAL_LLM, TOP_K, EMBED_MODEL, CHROMA_DIR, APP_TITLE
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
load_dotenv()

# instead of loading in local model, using api to access model since my pc doesn't have enough cpu/gpu to work it

hf = os.getenv("HF_TOKEN")

@st.cache_resource(show_spinner=False)
def get_inference_client():
    return InferenceClient(token=hf)

@st.cache_resource(show_spinner="Loading collection...")
def get_collection():
    """
    gets the chroma collection with embedding funct; cached so it doesn't reload every time
    """
    embedding_fn = ef.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION, embedding_function=embedding_fn)


def query_llm(messages):
    """ queries and returns response from the model, returning content of first choice"""
    client = get_inference_client()
    response = client.chat_completion(
        model=LOCAL_LLM, 
        messages=messages,
        max_tokens=512
    )
    return response.choices[0].message.content


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

if "history" not in st.session_state:
    st.session_state.history = []

# st.write("Model loaded:", "tokenizer" in globals())



collection = get_collection()

with st.form("qa_form"):
    question = st.text_input("Ask a question about the documents:")
    submitted = st.form_submit_button("Ask")

if submitted and question:
    # query collection for relevant chunks
    res = collection.query(query_texts=[question], n_results=TOP_K)
    chunks = list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))
    context = "\n\n---\n\n".join(c[0] for c in chunks)
    
    # gotta make that prompt, even if it's wacked up (only if it's out of context)
    messages = [
    {"role": "system", "content": "Answer using the context with enough detail and explanation. If not in context, be as unhinged as possible, using Gen Alpha slang; you can also add emojis or asciis that fit the brainrot feel. Do not add extra details. Do not speculate. Do not invent examples."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    reply = query_llm(messages)
    reply = reply.strip()

    st.markdown(f"### Answer\n{reply}")
    # st.write(reply)
    st.session_state.history.append({
        "q": question,
        "a": reply,
        "chunks": chunks
    })    



# this shows the history of questions and answers + sources
for idx, item in enumerate(reversed(st.session_state.history)):
    # added data viz to show the distances of retrieved chunks (originally for extra credit)
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
        st.plotly_chart(fig, width="stretch")

    st.markdown(f"**Q:** {item['q']}")
    st.markdown(f"**A:** {item['a']}")
    with st.expander("Sources"):
        for txt, meta, dist in item["chunks"]:
            st.write(f"- *{meta['source']}* (dist={dist:.3f})\n> {txt[:300]}…")