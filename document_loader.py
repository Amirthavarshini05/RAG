import fitz  # PyMuPDF

def load_pdf(file_path):
    doc = fitz.open(file_path)
    texts = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        texts.append((page_num, text))
    return texts
