from pathlib import Path
import fitz

pages = []

pdf_folder = Path("pdfs")
pdf_files = pdf_folder.glob("*.pdf")

for pdf_file in pdf_files:
    pdf = fitz.open(pdf_file)

    for page_number, page in enumerate(pdf):
        text = page.get_text()

        pages.append({
            "document": pdf_file.name,
            "page": page_number + 1,
            "text": text
        })

for page in pages:
    print("Document:", page["document"])
    print("Page:", page["page"])
    print("Text:", page["text"])
    print("-" * 50)