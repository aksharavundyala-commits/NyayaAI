import os
import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_ollama import (
    ChatOllama,
    OllamaEmbeddings
)

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="NyayaAI",
    page_icon="⚖️",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("⚖️ NyayaAI")

st.subheader(
    "AI-Powered Legal Assistance and Document Analysis System"
)

st.info(
    "NyayaAI is an educational legal-document "
    "analysis system and does not provide professional legal advice."
)


# ==========================================
# PROJECT DIRECTORIES
# ==========================================

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"

DB_DIR = BASE_DIR / "chroma_db"

DATA_DIR.mkdir(exist_ok=True)


# ==========================================
# OLLAMA CONFIGURATION
# ==========================================

OLLAMA_MODEL = "llama3.2"

EMBEDDING_MODEL = "nomic-embed-text"


# ==========================================
# LOAD OLLAMA MODELS
# ==========================================

@st.cache_resource
def load_models():

    # Chat model

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0
    )

    # Embedding model

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    return llm, embeddings


# ==========================================
# LOAD PDF / DOCX
# ==========================================

def load_document(file_path):

    extension = file_path.suffix.lower()

    if extension == ".pdf":

        loader = PyPDFLoader(
            str(file_path)
        )

        return loader.load()

    elif extension == ".docx":

        loader = Docx2txtLoader(
            str(file_path)
        )

        return loader.load()

    return []


# ==========================================
# CREATE VECTOR DATABASE
# ==========================================

def create_vector_database(documents):

    # Split document

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(
        documents
    )

    # Load Ollama embeddings

    llm, embeddings = load_models()

    # Create Chroma database

    vector_database = Chroma(
        collection_name="nyayaai_documents",
        embedding_function=embeddings,
        persist_directory=str(DB_DIR)
    )

    # Store chunks

    vector_database.add_documents(
        chunks
    )

    return vector_database, len(chunks)


# ==========================================
# GET VECTOR DATABASE
# ==========================================

def get_vector_database():

    llm, embeddings = load_models()

    vector_database = Chroma(
        collection_name="nyayaai_documents",
        embedding_function=embeddings,
        persist_directory=str(DB_DIR)
    )

    return vector_database


# ==========================================
# UPLOAD DOCUMENT
# ==========================================

uploaded_files = st.file_uploader(
    "Upload Legal Documents",
    type=["pdf", "docx"],
    accept_multiple_files=True
)


# ==========================================
# PROCESS DOCUMENTS
# ==========================================

if uploaded_files:

    if st.button("Process Documents"):

        all_documents = []

        for uploaded_file in uploaded_files:

            file_path = (
                DATA_DIR /
                uploaded_file.name
            )

            # Save uploaded document

            with open(
                file_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )

            # Extract document text

            documents = load_document(
                file_path
            )

            all_documents.extend(
                documents
            )


        if all_documents:

            database, number_of_chunks = (
                create_vector_database(
                    all_documents
                )
            )

            st.success(
                "Documents processed successfully!"
            )

            st.write(
                f"Created {number_of_chunks} text chunks."
            )

            st.session_state[
                "documents_ready"
            ] = True

        else:

            st.error(
                "Could not read the documents."
            )


# ==========================================
# CHECK DATABASE
# ==========================================

if DB_DIR.exists():

    st.session_state[
        "documents_ready"
    ] = True


st.divider()


# ==========================================
# QUESTION
# ==========================================

question = st.text_input(
    "Ask a question about your legal document",
    placeholder=(
        "Example: What are the main obligations "
        "mentioned in this agreement?"
    )
)


# ==========================================
# QUESTION ANSWERING
# ==========================================

if question:

    if not st.session_state.get(
        "documents_ready",
        False
    ):

        st.warning(
            "Please upload and process a document first."
        )

    else:

        try:

            # Get Chroma database

            database = get_vector_database()


            # Search relevant chunks

            relevant_documents = (
                database.similarity_search(
                    question,
                    k=4
                )
            )


            if not relevant_documents:

                st.warning(
                    "No relevant information found."
                )


            else:

                # Create context

                context = "\n\n".join(

                    [
                        f"Source: "
                        f"{document.metadata.get('source', 'Unknown')}\n"
                        f"{document.page_content}"

                        for document
                        in relevant_documents
                    ]
                )


                # ==================================
                # PROMPT
                # ==================================

                prompt = ChatPromptTemplate.from_template(
                    """
You are NyayaAI, an educational
legal document analysis assistant.

Use ONLY the information provided
in the document context.

Do not invent:

- Laws
- Legal sections
- Facts
- Cases
- Citations
- Legal conclusions

If the answer is not available
in the uploaded document, say:

"The information is not available
in the uploaded document."

Explain complicated legal language
in simple terms.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""
                )


                # Load Ollama

                llm, embeddings = load_models()


                # Format prompt

                messages = prompt.format_messages(
                    context=context,
                    question=question
                )


                # Ask Ollama

                response = llm.invoke(
                    messages
                )


                # ==================================
                # DISPLAY ANSWER
                # ==================================

                st.markdown(
                    "### ⚖️ NyayaAI Answer"
                )

                st.write(
                    response.content
                )


                # ==================================
                # SOURCES
                # ==================================

                st.markdown(
                    "### 📄 Retrieved Sources"
                )


                for index, document in enumerate(
                    relevant_documents,
                    start=1
                ):

                    source = Path(
                        document.metadata.get(
                            "source",
                            "Unknown"
                        )
                    ).name

                    page = document.metadata.get(
                        "page"
                    )

                    if isinstance(page, int):

                        st.write(
                            f"{index}. {source} "
                            f"(Page {page + 1})"
                        )

                    else:

                        st.write(
                            f"{index}. {source}"
                        )


        except Exception as error:

            st.error(
                f"Error: {error}"
            )


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚖️ NyayaAI")

    st.write(
        """
NyayaAI uses Retrieval-Augmented
Generation (RAG) to analyze
legal documents.
"""
    )

    st.markdown("### RAG Pipeline")

    st.write("1. Upload PDF/DOCX")
    st.write("2. Extract text")
    st.write("3. Split text")
    st.write("4. Ollama Embeddings")
    st.write("5. Chroma Database")
    st.write("6. Similarity Search")
    st.write("7. Ollama LLM")
    st.write("8. Generate Answer")