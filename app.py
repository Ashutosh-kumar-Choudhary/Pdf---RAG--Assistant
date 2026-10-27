import streamlit as st
import ollama
import chromadb
import fitz


# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="pdf_collection"
)


# Page configuration
st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="📄"
)

st.title("📄 PDF Question Answering")
st.write("Upload a PDF and ask questions about it.")


# PDF upload
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# Process PDF only when a new PDF is uploaded
if uploaded_file is not None:

    if "processed_file" not in st.session_state or \
       st.session_state.processed_file != uploaded_file.name:

        with st.spinner("Processing PDF..."):

            pdf = fitz.open(
                stream=uploaded_file.getvalue(),
                filetype="pdf"
            )

            chunks = []

            for page_number, page in enumerate(pdf):

                text = page.get_text()

                words = text.split()

                chunk_size = 100

                for i in range(0, len(words), chunk_size):

                    chunk_words = words[i:i + chunk_size]

                    chunk_text = " ".join(chunk_words)

                    chunks.append({
                        "document": uploaded_file.name,
                        "page": page_number + 1,
                        "text": chunk_text
                    })


            # Store embeddings
            for index, chunk in enumerate(chunks):

                response = ollama.embed(
                    model="nomic-embed-text",
                    input=chunk["text"]
                )

                embedding = response["embeddings"][0]

                collection.upsert(
                    ids=[
                        f"{uploaded_file.name}{chunk['page']}{index}"
                    ],
                    embeddings=[embedding],
                    documents=[chunk["text"]],
                    metadatas=[{
                        "document": chunk["document"],
                        "page": chunk["page"]
                    }]
                )


            st.session_state.processed_file = uploaded_file.name

        st.success("PDF processed successfully!")


    # Question section
    question = st.text_input(
        "Ask a question about the PDF:"
    )


    if st.button("Ask Question"):

        if question:

            with st.spinner("Finding the answer..."):

                # Embed question
                response = ollama.embed(
                    model="nomic-embed-text",
                    input=question
                )

                question_embedding = response["embeddings"][0]


                # Retrieve relevant chunks
                results = collection.query(
                    query_embeddings=[question_embedding],
                    n_results=3
                )

                relevant_chunks = results["documents"][0]
                metadata = results["metadatas"][0]


                # Create context
                context = "\n\n".join(relevant_chunks)


                # Ask Qwen3
                prompt = f"""
Answer the question using only the information in the context below.

Context:
{context}

Question:
{question}

If the answer cannot be found in the context, say:
"I could not find the answer in the document."
"""


                answer = ollama.chat(
                    model="qwen3:4b",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    think=False
                )


            st.subheader("Answer")
            st.write(answer.message.content)


            st.subheader("Sources")

            for source in metadata:

                st.write(
                    f"- {source['document']} — Page {source['page']}"
                )

        else:

            st.warning("Please enter a question.")