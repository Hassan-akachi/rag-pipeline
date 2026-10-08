# ============================================================
# Liverpool FC RAG (Retrieval-Augmented Generation) App
# ------------------------------------------------------------
# Flow: liverpool_facts.csv -> embeddings -> ChromaDB
#       -> find similar facts -> add them to the prompt
#       -> LLM (Ollama) answers using only those facts
# Run with:  streamlit run rag_liverpool.py
# Needs liverpool_facts.csv in the same folder.
# ============================================================

import os
import streamlit as st  # Web UI framework
import pandas as pd  # Reads the CSV file into a table
import chromadb  # Vector database that stores embeddings
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions  # Built-in embedding helpers
from openai import OpenAI  # OpenAI-style client, used here to talk to Ollama
from google import genai  # Google Gemini SDK (package: google-genai)
from google.genai import types
from dotenv import load_dotenv  # Loads API keys from a .env file

# Stop HuggingFace tokenizer warnings (used by Chroma's default model)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Read GEMINI_API_KEY from the .env file
load_dotenv()

# Path to the knowledge base (CSV must have columns: id, category, fact)
CSV_FILE = "liverpool_facts.csv"


# ------------------------------------------------------------
# Custom Gemini embedding function for ChromaDB
# ------------------------------------------------------------
class GeminiEmbeddingFunction(EmbeddingFunction):
    """Turns a list of texts into vectors using Google's Gemini API."""

    def __init__(self, api_key, model_name="gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)  # Connect to Gemini
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        # One request for all texts -> one vector per text
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=input,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        return [e.values for e in result.embeddings]


# ------------------------------------------------------------
# Embedding model selector (OpenAI removed: no credits)
# ------------------------------------------------------------
class EmbeddingModel:
    def __init__(self, model_type="gemini"):
        self.model_type = model_type

        if model_type == "gemini":
            # Google Gemini embeddings (needs GEMINI_API_KEY)
            self.embedding_fn = GeminiEmbeddingFunction(
                api_key=os.getenv("GEMINI_API_KEY")
            )
        elif model_type == "chroma":
            # Free local model (all-MiniLM-L6-v2), downloads on first use
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        elif model_type == "nomic":
            # Local Ollama embedding model via Ollama's OpenAI-style API
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key="ollama",  # Ollama ignores the key, but one is required
                api_base="http://localhost:11434/v1",
                model_name="nomic-embed-text",
            )
        else:
            # Clear error instead of a confusing AttributeError later
            raise ValueError(f"Embedding model '{model_type}' is not available")


# ------------------------------------------------------------
# LLM: the model that writes the final answer (local Ollama)
# ------------------------------------------------------------
class LLMModel:
    def __init__(self, model_name="llama3.2"):
        self.model_type = "ollama"
        self.model_name = model_name
        # Ollama must be running locally (ollama serve)
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    def generate_completion(self, messages):
        """Send chat messages to the LLM and return its reply text."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,  # Low = sticks closer to the facts
            )
            return response.choices[0].message.content
        except Exception as e:
            # Show the error in the UI instead of crashing
            return f"Error generating response: {str(e)}"


# ------------------------------------------------------------
# Load the knowledge base from CSV
# ------------------------------------------------------------
def load_facts(csv_path=CSV_FILE):
    """Read the CSV file into a pandas DataFrame."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["fact"])  # Skip empty rows
    return df


# ------------------------------------------------------------
# Vector database setup
# ------------------------------------------------------------
def setup_chromadb(df, embedding_model):
    """Store every fact (plus its category) in an in-memory Chroma collection."""
    client = chromadb.Client()  # In-memory DB (rebuilt each run)

    # Remove any old collection: different embedding models
    # make vectors of different sizes, which can't be mixed
    try:
        client.delete_collection("liverpool_facts")
    except Exception:
        pass  # Didn't exist yet — fine

    collection = client.create_collection(
        name="liverpool_facts", embedding_function=embedding_model.embedding_fn
    )

    # documents = the text to search; metadatas = extra info we show later
    collection.add(
        documents=df["fact"].tolist(),
        metadatas=[{"category": c} for c in df["category"]],
        ids=[str(i) for i in df["id"]],
    )
    return collection


# ------------------------------------------------------------
# RAG steps
# ------------------------------------------------------------
def find_related_chunks(query, collection, top_k=3):
    """Retrieve: get the top_k facts closest in meaning to the question."""
    results = collection.query(query_texts=[query], n_results=top_k)
    # Pair each fact with its metadata, e.g. ("Ian Rush ...", {"category": "Players"})
    return list(zip(results["documents"][0], results["metadatas"][0]))


def augment_prompt(query, related_chunks):
    """Augment: put the retrieved facts in front of the question."""
    context = "\n".join(f"- {doc}" for doc, _ in related_chunks)
    return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"


def rag_pipeline(query, collection, llm_model, top_k=3):
    """Full RAG: retrieve -> augment -> generate."""
    related_chunks = find_related_chunks(query, collection, top_k)  # 1. Retrieve
    augmented_prompt = augment_prompt(query, related_chunks)  # 2. Augment

    # 3. Generate: system message keeps answers grounded in the context
    response = llm_model.generate_completion(
        [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant who answers questions about "
                    "Liverpool Football Club. Only use the facts in the context. "
                    "If the answer is not in the context, say you don't know."
                ),
            },
            {"role": "user", "content": augmented_prompt},
        ]
    )
    return response, related_chunks, augmented_prompt


# ------------------------------------------------------------
# Streamlit user interface
# ------------------------------------------------------------
def streamlit_app():
    st.set_page_config(page_title="Liverpool FC RAG", page_icon="🔴", layout="wide")
    st.title("🔴 Liverpool FC RAG System")

    # --- Sidebar: model choices ---
    st.sidebar.title("Model Configuration")
    st.sidebar.write("LLM: Ollama Llama 3.2 (local)")

    embedding_type = st.sidebar.radio(
        "Select Embedding Model:",
        ["gemini", "chroma", "nomic"],
        format_func=lambda x: {
            "gemini": "Google Gemini Embeddings",
            "chroma": "Chroma Default (local)",
            "nomic": "Nomic Embed Text (Ollama)",
        }[x],
    )

    # How many facts to retrieve for each question
    top_k = st.sidebar.slider("Facts to retrieve (top_k)", 1, 5, 3)

    # --- First run: load CSV, create models and database ---
    # st.session_state keeps values between Streamlit reruns
    if "initialized" not in st.session_state:
        st.session_state.facts_df = load_facts()
        st.session_state.llm_model = LLMModel()
        st.session_state.embedding_model = EmbeddingModel(embedding_type)
        st.session_state.collection = setup_chromadb(
            st.session_state.facts_df, st.session_state.embedding_model
        )
        st.session_state.initialized = True

    # --- Embedding model changed? Rebuild the vector database ---
    if st.session_state.embedding_model.model_type != embedding_type:
        st.session_state.embedding_model = EmbeddingModel(embedding_type)
        st.session_state.collection = setup_chromadb(
            st.session_state.facts_df, st.session_state.embedding_model
        )

    # --- Show the knowledge base as a table ---
    with st.expander("📚 Liverpool Facts (from CSV)", expanded=False):
        st.dataframe(st.session_state.facts_df, hide_index=True, use_container_width=True)

    # --- Question box ---
    query = st.text_input(
        "Ask a question about Liverpool FC:",
        placeholder="e.g., Who is Liverpool's all-time top scorer?",
    )

    # --- Run RAG when a question is entered ---
    if query:
        with st.spinner("Searching the Kop for answers..."):
            response, chunks, augmented_prompt = rag_pipeline(
                query, st.session_state.collection, st.session_state.llm_model, top_k
            )

        # Answer on the left, sources on the right
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🤖 Response")
            st.write(response)
        with col2:
            st.markdown("### 📖 References Used")
            for doc, meta in chunks:
                st.write(f"- **[{meta.get('category', '')}]** {doc}")

        # Behind-the-scenes info for learning/debugging
        with st.expander("🔍 Technical Details", expanded=False):
            st.markdown("#### Augmented Prompt")
            st.code(augmented_prompt)
            st.markdown("#### Model Configuration")
            st.write("- LLM Model: OLLAMA (llama3.2)")
            st.write(f"- Embedding Model: {embedding_type.upper()}")


# Start the app when run directly
if __name__ == "__main__":
    streamlit_app()
