# ============================================================
# Space Facts RAG (Retrieval-Augmented Generation) App
# ------------------------------------------------------------
# Flow: facts -> embeddings -> ChromaDB -> find similar facts
#       -> add them to the prompt -> LLM answers using them
# Run with:  streamlit run rag-streamlit.py
# ============================================================

import streamlit as st  # Web UI framework
import csv  # To save the facts to a CSV file
import pandas as pd  # (Optional) handy for working with tabular data
import chromadb  # Vector database that stores embeddings
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions  # Built-in embedding helpers
from openai import OpenAI  # OpenAI client (also works with Ollama's API)
from google import genai  # Google Gemini SDK (package: google-genai)
from google.genai import types
import os
from dotenv import load_dotenv  # Loads API keys from a .env file


# Stop HuggingFace tokenizer warnings (used by Chroma's default model)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Read OPENAI_API_KEY / GEMINI_API_KEY from the .env file
load_dotenv()


# ------------------------------------------------------------
# Custom Gemini embedding function for ChromaDB
# ------------------------------------------------------------
class GeminiEmbeddingFunction(EmbeddingFunction):
    """Turns a list of texts into vectors using Google's Gemini API."""

    def __init__(self, api_key, model_name="gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)  # Connect to Gemini
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        # Send all texts in one request and get one vector per text back
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=input,
            # SEMANTIC_SIMILARITY works for both stored facts and user questions
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        return [e.values for e in result.embeddings]


# ------------------------------------------------------------
# Embedding model selector
# Embeddings convert text into numbers so we can compare meaning.
# ------------------------------------------------------------
class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type

        # if model_type == "openai":
        #     # OpenAI's hosted embedding model (needs OPENAI_API_KEY)
        #     self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        #         api_key=os.getenv("OPENAI_API_KEY"),
        #         model_name="text-embedding-3-small",
        #     )

        if model_type == "chroma":
            # Free local model (all-MiniLM-L6-v2), downloads on first use
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        elif model_type == "nomic":
            # Local Ollama model, accessed through Ollama's OpenAI-style API
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key="ollama",  # Ollama ignores the key, but one is required
                api_base="http://localhost:11434/v1",
                model_name="nomic-embed-text",
            )

        elif model_type == "gemini":
            # Google Gemini embeddings (needs GEMINI_API_KEY)
            self.embedding_fn = GeminiEmbeddingFunction(
                api_key=os.getenv("GEMINI_API_KEY")
            )
        else:
            raise ValueError(f"Embedding model '{model_type}' is not available")

# ------------------------------------------------------------
# LLM selector: the model that writes the final answer
# ------------------------------------------------------------
class LLMModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
         # OpenAI is disabled because the API account currently has no credits.
        # if model_type == "openai":
        #     self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        #     self.model_name = "gpt-4o-mini"
        # else:
            # Local Ollama model (Ollama must be running)
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.model_name = "llama3.2"

    def generate_completion(self, messages):
        """Send the chat messages to the LLM and return its reply text."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,  # Higher = more creative, lower = more factual
            )
            return response.choices[0].message.content
        except Exception as e:
            # Show the error in the UI instead of crashing the app
            return f"Error generating response: {str(e)}"


# ------------------------------------------------------------
# Knowledge base: our "documents"
# ------------------------------------------------------------
def generate_csv():
    """Create the list of space facts and save a copy to space_facts.csv."""
    facts = [
        {"id": 1, "fact": "The first human to orbit Earth was Yuri Gagarin in 1961."},
        {"id": 2, "fact": "The Apollo 11 mission landed the first humans on the Moon in 1969."},
        {"id": 3, "fact": "The Hubble Space Telescope was launched in 1990 and has provided stunning images of the universe."},
        {"id": 4, "fact": "Mars is the most explored planet in the solar system, with multiple rovers sent by NASA."},
        {"id": 5, "fact": "The International Space Station (ISS) has been continuously occupied since November 2000."},
        {"id": 6, "fact": "Voyager 1 is the farthest human-made object from Earth, launched in 1977."},
        {"id": 7, "fact": "SpaceX, founded by Elon Musk, is the first private company to send humans to orbit."},
        {"id": 8, "fact": "The James Webb Space Telescope, launched in 2021, is the successor to the Hubble Telescope."},
        {"id": 9, "fact": "The Milky Way galaxy contains over 100 billion stars."},
        {"id": 10, "fact": "Black holes are regions of spacetime where gravity is so strong that nothing can escape."},
    ]

    # Write the facts to a CSV file (useful for inspection / reuse)
    with open("space_facts.csv", mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "fact"])
        writer.writeheader()
        writer.writerows(facts)
    return facts


# ------------------------------------------------------------
# Vector database setup
# ------------------------------------------------------------
def setup_chromadb(documents, embedding_model):
    """Store every fact as an embedding in an in-memory Chroma collection."""
    client = chromadb.Client()  # In-memory DB (data is lost on restart)

    # Delete the old collection, since different embedding models
    # produce vectors of different sizes and can't be mixed
    try:
        client.delete_collection("space_facts")
    except Exception:
        pass  # Collection didn't exist yet — that's fine

    collection = client.create_collection(
        name="space_facts", embedding_function=embedding_model.embedding_fn
    )

    # Chroma embeds each document automatically using the function above
    collection.add(documents=documents, ids=[str(i) for i in range(len(documents))])

    return collection


# ------------------------------------------------------------
# RAG steps
# ------------------------------------------------------------
def find_related_chunks(query, collection, top_k=2):
    """Retrieve: find the top_k facts whose meaning is closest to the query."""
    results = collection.query(query_texts=[query], n_results=top_k)
    # Pair each document with its metadata (empty dicts if none were stored)
    return list(
        zip(
            results["documents"][0],
            (
                results["metadatas"][0]
                if results["metadatas"][0]
                else [{}] * len(results["documents"][0])
            ),
        )
    )


def augment_prompt(query, related_chunks):
    """Augment: put the retrieved facts in front of the question."""
    context = "\n".join([chunk[0] for chunk in related_chunks])
    return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"


def rag_pipeline(query, collection, llm_model, top_k=2):
    """Full RAG: retrieve -> augment -> generate."""
    related_chunks = find_related_chunks(query, collection, top_k)  # 1. Retrieve
    augmented_prompt = augment_prompt(query, related_chunks)  # 2. Augment

    # 3. Generate: the system message keeps the LLM grounded in the context
    response = llm_model.generate_completion(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant who can answer questions about space but only answers questions that are directly related to the sources/documents given.",
            },
            {"role": "user", "content": augmented_prompt},
        ]
    )

    references = [chunk[0] for chunk in related_chunks]  # Facts that were used
    return response, references, augmented_prompt


# ------------------------------------------------------------
# Streamlit user interface
# ------------------------------------------------------------
def streamlit_app():
    st.set_page_config(page_title="Space Facts RAG", layout="wide")
    st.title("🚀 Space Facts RAG System")

    # --- Sidebar: choose which models to use ---
    st.sidebar.title("Model Configuration")

    llm_type = st.sidebar.radio(
    "Select LLM Model:",
    ["ollama"],  # "openai" removed (no credits)
    format_func=lambda x: "Ollama Llama 3.2",
)

    embedding_type = st.sidebar.radio(
        "Select Embedding Model:",
        ["chroma", "nomic", "gemini"],
        format_func=lambda x: {
            # "openai": "OpenAI Embeddings",
            "chroma": "Chroma Default",
            "nomic": "Nomic Embed Text (Ollama)",
            "gemini": "Google Gemini Embeddings",
        }[x],
    )

    # --- First run only: build facts, models and database ---
    # st.session_state keeps values between Streamlit reruns
    if "initialized" not in st.session_state:
        st.session_state.facts = generate_csv()
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)

        documents = [fact["fact"] for fact in st.session_state.facts]
        st.session_state.collection = setup_chromadb(
            documents, st.session_state.embedding_model
        )
        st.session_state.initialized = True

    # --- If the LLM changed, just swap it ---
    if st.session_state.llm_model.model_type != llm_type:
        st.session_state.llm_model = LLMModel(llm_type)

    # --- If the embedding model changed, rebuild the vector database ---
    if st.session_state.embedding_model.model_type != embedding_type:
        st.session_state.embedding_model = EmbeddingModel(embedding_type)
        documents = [fact["fact"] for fact in st.session_state.facts]
        st.session_state.collection = setup_chromadb(
            documents, st.session_state.embedding_model
        )

    # --- Show the knowledge base ---
    with st.expander("📚 Available Space Facts", expanded=False):
        for fact in st.session_state.facts:
            st.write(f"- {fact['fact']}")

    # --- Question box ---
    query = st.text_input(
        "Enter your question about space:",
        placeholder="e.g., What is the Hubble Space Telescope?",
    )

    # --- Run RAG when the user types a question ---
    if query:
        with st.spinner("Processing your query..."):
            response, references, augmented_prompt = rag_pipeline(
                query, st.session_state.collection, st.session_state.llm_model
            )

            # Answer on the left, sources on the right
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🤖 Response")
                st.write(response)

            with col2:
                st.markdown("### 📖 References Used")
                for ref in references:
                    st.write(f"- {ref}")

            # Behind-the-scenes info for learning/debugging
            with st.expander("🔍 Technical Details", expanded=False):
                st.markdown("#### Augmented Prompt")
                st.code(augmented_prompt)

                st.markdown("#### Model Configuration")
                st.write(f"- LLM Model: {llm_type.upper()}")
                st.write(f"- Embedding Model: {embedding_type.upper()}")


# Start the app when run directly
if __name__ == "__main__":
    streamlit_app()