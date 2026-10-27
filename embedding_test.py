import ollama
import chromadb
from chunk_test import chunks

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="pdf_collection"
)

for chunk in chunks:
    response = ollama.embed(
        model="nomic-embed-text",
        input=chunk["text"]
    )

    embedding = response["embeddings"][0]

    collection.add(
        ids=[f'{chunk["document"]}{chunk["page"]}{chunks.index(chunk)}'],
        embeddings=[embedding],
        documents=[chunk["text"]],
        metadatas=[{
            "document": chunk["document"],
            "page": chunk["page"]
        }]
    )

print("Embeddings stored successfully!")