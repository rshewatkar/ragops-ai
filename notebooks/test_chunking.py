import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.rag_chain import load_and_chunk_pdf

chunks = load_and_chunk_pdf("data\Rahul_Shewatkar_Resume.pdf")

for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---\n")
    print(chunk.page_content)