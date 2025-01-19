import chromadb
import numpy as np

# Initialize ChromaDB client
client = chromadb.Client()

# Create or access a collection in ChromaDB
collection = client.get_or_create_collection(name="text_embeddings")

# Simulate text data (you can replace this with any text)
text = """This is the first line.
This is the second line.
This is the third line."""

# Split the text into lines
lines = text.split("\n")

# Simulate embedding generation (replace with real embedding function)
def generate_embedding(text: str):
    # Simulating an embedding as a random vector of size 5 (for simplicity)
    return np.random.rand(5).tolist()

# Store each line and its embedding in ChromaDB
for idx, line in enumerate(lines):
    embedding = generate_embedding(line)
    
    # Debug: Print the line and the generated embedding
    print(f"Adding line {idx}: {line} with embedding: {embedding}")
    
    # Add the document and its embedding to ChromaDB
    collection.add(
        documents=[line],  # List containing the line (document)
        embeddings=[embedding],  # List containing the generated embedding
        ids=[str(idx)],  # Unique ID for each document
        metadatas=[{"line_number": idx + 1}]  # Metadata (optional)
    )

# Retrieve the first document using ID '0'
result = collection.get(ids=["0"])

# Debug print the result to understand its structure
print("Query Result:", result)

# Check if embeddings are present before trying to access them
if result['embeddings'] is not None:
    print("Text:", result['documents'][0])
    print("Embedding:", result['embeddings'][0])
else:
    print("No embeddings found for the requested document.")

x=collection.get(ids=["0"])
print(x)
