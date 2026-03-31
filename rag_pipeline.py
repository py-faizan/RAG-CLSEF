# ✅ Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
SAVE_DIR = "/content/drive/MyDrive/v3_rag_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ Install required packages
!pip install -q google-generativeai PyMuPDF faiss-cpu sentence-transformers pdf2image PyPDF2
# ✅ Install doctr with all necessary OCR dependencies
!pip install 'python-doctr[torch,tf]' --upgrade

!apt-get install -y poppler-utils

# ✅ Define PDF path
pdf_path = "/content/drive/MyDrive/Colab Notebooks/Prospectus-2025-v1.pdf"
ocr_text_path = os.path.join(SAVE_DIR, "extracted_text_doctr.txt")
print(ocr_text_path)

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from tqdm import tqdm
import os

def extract_text_and_tables(pdf_path, save_path, images_dir, dpi=200):
    os.makedirs(images_dir, exist_ok=True)

    # If already processed, load cached result
    if os.path.exists(save_path):
        print("✅ Loaded cached OCR text.")
        with open(save_path, "r", encoding="utf-8") as f:
            return f.read()

    # Count total pages
    total_pages = len(PdfReader(pdf_path).pages)

    # Load OCR model ONCE
    model = ocr_predictor(pretrained=True)

    full_text = ""
    print(f"🔄 Processing {total_pages} pages...")

    for page_num in tqdm(range(total_pages), desc="📄 OCR Progress"):
        # Convert current page to image
        images = convert_from_path(
            pdf_path, dpi=dpi,
            first_page=page_num+1, last_page=page_num+1
        )
        if not images:
            continue

        img_filename = f"page_{page_num+1}.png"
        img_path = os.path.join(images_dir, img_filename)
        images[0].save(img_path)

        # OCR this image
        doc = DocumentFile.from_images([img_path])
        result = model(doc)
        page_data = result.export()["pages"]

        if page_data:
            page_text = ""
            for block in page_data[0].get("blocks", []):
                if "lines" in block:  # Normal text block
                    for line in block["lines"]:
                        line_text = " ".join([w["value"] for w in line["words"]])
                        page_text += line_text + "\n"

                elif "table" in block:  # Table detected
                    table_data = block["table"]["cells"]
                    for row in table_data:
                        row_text = "\t".join([cell["content"] for cell in row])
                        page_text += row_text + "\n"

            full_text += f"\n\n--- Page {page_num+1} ---\n{page_text.strip()}"

    # Save final text
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"✅ OCR completed. Total pages processed: {total_pages}")
    return full_text


images_folder = "/content/drive/MyDrive/v3_rag_data/extracted_images"

pdf_text = extract_text_and_tables(pdf_path, ocr_text_path, images_folder)

# ✅ Chunking
import pickle
chunks_path = os.path.join(SAVE_DIR, "chunks.pkl")

def chunk_text(text, max_words=400, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(' '.join(words[start:end]))
        if end == len(words):
            break
        start += max_words - overlap
    return chunks

if os.path.exists(chunks_path):
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print("✅ Loaded cached chunks.")
else:
    chunks = chunk_text(pdf_text)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print("✅ Chunks created and saved.")

print(f"🧩 Total Chunks: {len(chunks)}")

# ✅ Embedding
import numpy as np
from sentence_transformers import SentenceTransformer

embedding_path = os.path.join(SAVE_DIR, "embeddings.npy")
embedding_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

if os.path.exists(embedding_path):
    embeddings = np.load(embedding_path)
    print("✅ Loaded cached embeddings.")
else:
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    np.save(embedding_path, embeddings)
    print("✅ Embeddings created and saved.")

# ✅ FAISS Index
import faiss
faiss_index_path = os.path.join(SAVE_DIR, "faiss.index")
dimension = embeddings.shape[1]

if os.path.exists(faiss_index_path):
    index = faiss.read_index(faiss_index_path)
    print("✅ Loaded cached FAISS index.")
else:
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, faiss_index_path)
    print("✅ FAISS index created and saved.")


instruction = (
    # 1. Core Persona and Constraint
    "You are an expert Academic Policy Analyst for the university. Your ONLY source of information "
    "is the provided 'Context'. You MUST NOT use external or general knowledge. "

    # 2. Response Format and Tone
    "Answer professionally and concisely. Do not include phrases like 'Based on the provided text' "
    "or 'According to the context'. "


    # 3. Handling Specific Query Types (Crucial for your Categories)
    "If a question is unanswerable from the provided Context, you MUST reply with the exact phrase: "
    "'Sorry, I do not know the answer'. "

    "If the answer requires combining information from multiple facts/sections (synthesis), do so precisely "
    "and state ALL required details. "

    # 4. Handling Ambiguity and Multiple Facts
    "If two or more separate instances/facts are available for one query, reply with ALL of them clearly listed. "
    "DO NOT PARAPHRASE numerical data (e.g., fees, ages, percentages); quote them exactly. "
    "\n\n"
)

# ✅ Retrieval and LLM Response
query = "Who is the dean of faculty of physical sciences?"
query_embedding = embedding_model.encode([query], convert_to_numpy=True)

k_retrieval = 4
D, I = index.search(query_embedding, k=k_retrieval)
retrieved_chunks_text = [chunks[i] for i in I[0] if i != -1]
# "Start the response with 'Welcome to GCUF!'. "
context = instruction + "\n".join(retrieved_chunks_text)

final_prompt = f"""
You are an assistant helping answer questions about a university's academic policies and admissions.

Context:
{context}

Question:
{query}
"""

# print("🔍 Top Retrieved Chunks:")
# for i, chunk_text in enumerate(retrieved_chunks_text):
#     print(f"\n--- Chunk #{i+1} ---\n{chunk_text}...")

# ✅ Gemini LLM
import google.generativeai as genai
genai.configure(api_key="AIzaSyAVNId_0iM6nclZi9F2OSjXLrozNwmZYZw")

model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
response = model.generate_content(final_prompt)
# print("\n🧠 Gemini Response:\n")
print(response.text)
