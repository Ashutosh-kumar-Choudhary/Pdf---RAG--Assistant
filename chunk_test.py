from pdf_reader import pages

chunks = []

chunk_size = 100

for page in pages:
    words = page["text"].split()

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "document": page["document"],
            "page": page["page"],
            "text": chunk_text
        })