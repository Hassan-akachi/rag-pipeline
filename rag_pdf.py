# ============================================================
# PDF RAG (Retrieval-Augmented Generation) App
# ------------------------------------------------------------
# Flow: upload PDF -> extract text -> split into chunks
#       -> embed chunks -> store in ChromaDB (saved to disk)
#       -> find chunks similar to the question -> LLM answers
# Run with:  streamlit run rag_pdf.py
# ============================================================

import os
import uuid  # Creates unique IDs for each chunk
import streamlit as st  # Web UI framework
import chromadb  # Vector database
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions  # Built-in embedding helpers
from openai import OpenAI  # OpenAI-style client, used here to talk to Ollama
from google import genai  # Google Gemini SDK (package: google-genai)
from google.genai import types
from pypdf import PdfReader  # Reads PDF files (modern replacement for PyPDF2)
from dotenv import load_dotenv  # Loads API keys from a .env file

# Stop HuggingFace tokenizer warnings (used by Chroma's default model)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Read GEMINI_API_KEY (and OPENAI_API_KEY if you re-enable it) from .env
load_dotenv()

# Chunking settings (in characters)
CHUNK_SIZE = 1000  # Max length of each chunk
CHUNK_OVERLAP = 200  # Characters shared between neighbouring chunks


# ------------------------------------------------------------
# Custom Gemini embedding function for ChromaDB
# ------------------------------------------------------------
class GeminiEmbeddingFunction(EmbeddingFunction):
    """Turns a list of texts into vectors using Google's Gemini API."""

    def __init__(self, api_key, model_name="gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)  # Connect to Gemini
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        # Gemini accepts up to 100 texts per request, so send in batches
        for i in range(0, len(input), 100):
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=input[i : i + 100],
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
            )
            vectors.extend(e.values for e in result.embeddings)
        return vectors


# ------------------------------------------------------------
# Sidebar model picker
# ------------------------------------------------------------
class SimpleModelSelector:
    """Lists the available models and shows radio buttons for them."""

    def __init__(self):
        # LLMs that write the answer (OpenAI removed: no credits)
        self.llm_models = {
            # "openai": "GPT-4o mini",
            "ollama": "Llama 3.2 (Ollama)",
        }

        # Embedding models and their vector sizes (dimensions)
        self.embedding_models = {
            # "openai": {"name": "OpenAI Embeddings", "dimensions": 1536,
            #            "model_name": "text-embedding-3-small"},
            "gemini": {
                "name": "Google Gemini Embeddings",
                "dimensions": 3072,
                "model_name": "gemini-embedding-001",
            },
            "chroma": {"name": "Chroma Default", "dimensions": 384, "model_name": None},
            "nomic": {
                "name": "Nomic Embed Text (Ollama)",
                "dimensions": 768,
                "model_name": "nomic-embed-text",
            },
        }

    def select_models(self):
        """Show the radio buttons and return the user's choices."""
        st.sidebar.title("📚 Model Selection")

        llm = st.sidebar.radio(
            "Choose LLM Model:",
            options=list(self.llm_models.keys()),
            format_func=lambda x: self.llm_models[x],
        )

        embedding = st.sidebar.radio(
            "Choose Embedding Model:",
            options=list(self.embedding_models.keys()),
            format_func=lambda x: self.embedding_models[x]["name"],
        )

        return llm, embedding


# ------------------------------------------------------------
# PDF reading and chunking
# ------------------------------------------------------------
class SimplePDFProcessor:
    """Extracts text from a PDF and splits it into overlapping chunks."""

    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_pdf(self, pdf_file):
        """Return all the text in the PDF as one string."""
        reader = PdfReader(pdf_file)
        # extract_text() can return None for image-only pages, hence `or ""`
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    def create_chunks(self, text, source_name):
        """Split text into chunks of ~chunk_size, overlapping by chunk_overlap.

        Overlap keeps sentences that cross a boundary readable in both chunks.
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # Try to end on a full stop, but only if it's in the second half
            # of the chunk (so chunks don't become tiny)
            if end < text_len:
                last_period = text.rfind(".", start, end)
                if last_period > start + self.chunk_size // 2:
                    end = last_period + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "id": str(uuid.uuid4()),  # Unique ID for ChromaDB
                        "text": chunk_text,
                        "metadata": {"source": source_name},  # Which PDF it came from
                    }
                )

            if end >= text_len:
                break  # Reached the end of the document

            # Step back by the overlap, but always move forward at least 1 char
            # (prevents the infinite loop in the original version)
            start = max(end - self.chunk_overlap, start + 1)

        return chunks


# ------------------------------------------------------------
# The RAG system: database + embeddings + LLM
# ------------------------------------------------------------
class SimpleRAGSystem:
    def __init__(self, embedding_model="gemini", llm_model="ollama"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # PersistentClient saves the database to ./chroma_db on disk,
        # so your uploaded PDFs survive app restarts
        self.db = chromadb.PersistentClient(path="./chroma_db")

        self.setup_embedding_function()

        # LLM client (only Ollama for now)
        # if llm_model == "openai":
        #     self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

        self.collection = self.setup_collection()

    def setup_embedding_function(self):
        """Pick the embedding function for the chosen model."""
        if self.embedding_model == "gemini":
            self.embedding_fn = GeminiEmbeddingFunction(
                api_key=os.getenv("GEMINI_API_KEY")
            )
        elif self.embedding_model == "nomic":
            # Nomic via Ollama's OpenAI-compatible API
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key="ollama",
                api_base="http://localhost:11434/v1",
                model_name="nomic-embed-text",
            )
        elif self.embedding_model == "chroma":
            # Free local model (all-MiniLM-L6-v2)
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        else:
            raise ValueError(f"Embedding model '{self.embedding_model}' is not available")

    def setup_collection(self):
        """Get or create a collection for this embedding model.

        Each embedding model gets its own collection because their
        vectors have different sizes and can't be stored together.
        """
        collection_name = f"documents_{self.embedding_model}"
        return self.db.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"model": self.embedding_model},
        )

    def add_documents(self, chunks):
        """Embed the chunks and save them in ChromaDB."""
        try:
            self.collection.add(
                ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                metadatas=[c["metadata"] for c in chunks],
            )
            return True
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            return False

    def query_documents(self, query, n_results=3):
        """Return the n_results chunks most similar to the question."""
        try:
            return self.collection.query(query_texts=[query], n_results=n_results)
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return None

    def generate_response(self, query, context_chunks):
        """Ask the LLM to answer using only the retrieved chunks."""
        context = "\n\n---\n\n".join(context_chunks)  # Separate chunks clearly
        prompt = f"""Based on the following context, please answer the question.
If you can't find the answer in the context, say "I don't know".

Context:
{context}

Question: {query}

Answer:"""

        try:
            response = self.llm.chat.completions.create(
                model="llama3.2",  # "gpt-4o-mini" if you re-enable OpenAI
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Low = sticks closer to the document
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None

    # ⭐ UNIQUE FEATURE: Smart Suggested Questions
    def suggest_questions(self, text, n=3):
        """Read the start of the PDF and ask the LLM for n good questions.

        Helps users who don't know what to ask: they just click one.
        """
        sample = text[:3000]  # First ~3000 chars is enough to know the topic
        prompt = f"""Read this document excerpt and write {n} short, specific questions
a reader could ask about it. The answers must be in the document.
Return ONLY the questions, one per line, with no numbering.

Document:
{sample}"""
        try:
            response = self.llm.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            lines = response.choices[0].message.content.split("\n")
            # Clean up: drop empty lines, numbering like "1." or "-", keep questions
            questions = [l.strip().lstrip("0123456789.-*) ").strip() for l in lines]
            questions = [q for q in questions if q.endswith("?")]
            return questions[:n]
        except Exception:
            return []  # Feature is optional: fail quietly

    def get_embedding_info(self):
        """Name and vector size of the current embedding model."""
        info = SimpleModelSelector().embedding_models[self.embedding_model]
        return {
            "name": info["name"],
            "dimensions": info["dimensions"],
            "model": self.embedding_model,
            "chunks_stored": self.collection.count(),  # How many chunks are saved
        }


# ------------------------------------------------------------
# Streamlit user interface
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="PDF RAG", page_icon="🤖")
    st.title("🤖 Simple PDF RAG System")

    # --- Session state survives Streamlit reruns ---
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()  # PDFs already added
    if "current_embedding_model" not in st.session_state:
        st.session_state.current_embedding_model = None
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "suggested" not in st.session_state:
        st.session_state.suggested = []  # Suggested questions for the last PDF

    # --- Sidebar choices ---
    llm_model, embedding_model = SimpleModelSelector().select_models()

    # --- Embedding model changed? Start a fresh RAG system ---
    if embedding_model != st.session_state.current_embedding_model:
        st.session_state.processed_files.clear()
        st.session_state.current_embedding_model = embedding_model
        st.session_state.rag_system = None

    # --- Create the RAG system (once per embedding model) ---
    try:
        if st.session_state.rag_system is None:
            st.session_state.rag_system = SimpleRAGSystem(embedding_model, llm_model)

        info = st.session_state.rag_system.get_embedding_info()
        st.sidebar.info(
            f"Current Embedding Model:\n"
            f"- Name: {info['name']}\n"
            f"- Dimensions: {info['dimensions']}\n"
            f"- Chunks stored: {info['chunks_stored']}"
        )
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return

    # --- PDF upload ---
    pdf_file = st.file_uploader("Upload PDF", type="pdf")

    if pdf_file and pdf_file.name not in st.session_state.processed_files:
        processor = SimplePDFProcessor()
        with st.spinner("Processing PDF..."):
            try:
                text = processor.read_pdf(pdf_file)  # 1. Extract text
                if not text.strip():
                    st.error("No text found. The PDF may be scanned images only.")
                else:
                    chunks = processor.create_chunks(text, pdf_file.name)  # 2. Chunk
                    if st.session_state.rag_system.add_documents(chunks):  # 3. Store
                        st.session_state.processed_files.add(pdf_file.name)
                        st.success(f"Processed {pdf_file.name} into {len(chunks)} chunks")
                        # ⭐ Generate clickable suggested questions for this PDF
                        with st.spinner("Thinking of questions you could ask..."):
                            st.session_state.suggested = (
                                st.session_state.rag_system.suggest_questions(text)
                            )
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

    # --- Question box (works if anything is stored, even from a past run) ---
    if st.session_state.rag_system.collection.count() > 0:
        st.markdown("---")
        st.subheader("🔍 Query Your Documents")

        # ⭐ Suggested questions: clicking one fills the question box
        if st.session_state.suggested:
            st.markdown("**💡 Suggested questions** (click one):")
            for i, q in enumerate(st.session_state.suggested):
                st.button(
                    q,
                    key=f"suggest_{i}",
                    # on_click runs before the rerun, so the text box shows q
                    on_click=lambda q=q: st.session_state.update(query=q),
                )

        # key="query" links this box to st.session_state.query
        query = st.text_input("Ask a question:", key="query")

        if query:
            with st.spinner("Generating response..."):
                results = st.session_state.rag_system.query_documents(query)  # Retrieve

                if results and results["documents"] and results["documents"][0]:
                    response = st.session_state.rag_system.generate_response(
                        query, results["documents"][0]  # Augment + generate
                    )

                    if response:
                        st.markdown("### 📝 Answer:")
                        st.write(response)

                        # Show which passages the answer was based on
                        with st.expander("View Source Passages"):
                            docs = results["documents"][0]
                            metas = results["metadatas"][0]
                            for idx, (doc, meta) in enumerate(zip(docs, metas), 1):
                                st.markdown(f"**Passage {idx}** — _{meta.get('source', '')}_")
                                st.info(doc)
    else:
        st.info("👆 Please upload a PDF document to get started!")


if __name__ == "__main__":
    main()
