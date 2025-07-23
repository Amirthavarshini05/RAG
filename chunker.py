import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def chunk_text(text, chunk_size=4, overlap=1):
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

