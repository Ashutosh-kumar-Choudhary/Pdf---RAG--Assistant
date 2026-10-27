import ollama
import chromadb

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="pdf_collection"
)

question = input("Ask a question about the PDF: ")

# 1. Convert the question into a vector
response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

question_embedding = response["embeddings"][0]

# 2. Find the 3 most relevant PDF chunks
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

relevant_chunks = results["documents"][0]
metadata = results["metadatas"][0]

# 3. Combine the retrieved chunks into context
context = "\n\n".join(relevant_chunks)

# 4. Give the context + question to Qwen3
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

print("\nAnswer:")
print(answer.message.content)

# 5. Show the sources
print("\nSources:")

for source in metadata:
    print(f"- {source['document']} — Page {source['page']}")