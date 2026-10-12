import chromadb

# 1. Connect to your Chroma DB instance
# Adjust the path to where your Chroma database is stored locally
client = chromadb.PersistentClient(path="./chroma_db") # Replace with your actual database path

# 2. Access your specific collection
collection = client.get_collection(name="law_sections") # Replace with your actual collection name

# 3. Delete the records where the metadata field is empty
# IMPORTANT: Replace 'act_name' with the exact metadata key you used (e.g., 'source', 'act_title', 'document_name')
collection.delete(
    where={"act_name": ""} 
)

print("Empty act sections deleted successfully.")