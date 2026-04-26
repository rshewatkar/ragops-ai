from app.rag_chain import create_vector_store, search_query

# Step 1: Create DB (run once)
create_vector_store("data/Rahul_Shewatkar_Resume.pdf")

# Step 2: Search query
search_query("What are skil's section?")