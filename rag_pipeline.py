import os
from document_loader import load_pdf
from chunker import chunk_text
from embeddings import get_embedder, embed_text
from vectorstore import setup_qdrant, add_chunks_to_qdrant, search
from llm_model import load_llm, generate_answer

# 🧠 Load embedder and LLM
embedder = get_embedder()
tokenizer, model = load_llm()

# ✅ Build Knowledge Base (Run only once at start)
def build_knowledge_base():
    setup_qdrant()
    all_chunks = []

    for filename in os.listdir("documents"):
        file_path = os.path.join("documents", filename)
        if filename.endswith(".pdf"):
            pages = load_pdf(file_path)
        else:
            continue

        for page_num, text in pages:
            chunks = chunk_text(text)
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "embedding": embed_text(embedder, chunk),
                    "payload": {
                        "filename": filename,
                        "page": page_num,
                        "chunk_id": f"{filename}_pg{page_num}_ch{idx}",
                        "text": chunk,
                    }
                })

    add_chunks_to_qdrant(all_chunks)
    print(f"✅ {len(all_chunks)} chunks uploaded to Qdrant.")


# 💬 Get response to a user query
def get_response(query):
    query_embedding = embed_text(embedder, query)
    results = search(query_embedding, top_k=1)

    if not results:
        return "I couldn't find anything relevant in the documents."

    top = results[0]
    context = top.payload
    print("DEBUG payload keys:", context.keys())  # ✅ safe logging

    # Extract values safely
    text = context.get("text", "[Text missing]")
    filename = context.get("filename", "Unknown")
    page = context.get("page", "?")
    chunk_id = context.get("chunk_id", "unknown")

    context = top.payload['text']
    question = query
    answer = generate_answer(context, question, tokenizer, model)


    return f"Answer: {answer}\n\n[Source: {filename} | Page {page} | Chunk {chunk_id}]"



if __name__ == "__main__":
    build_knowledge_base()
    print(get_response("What is the document about?"))

