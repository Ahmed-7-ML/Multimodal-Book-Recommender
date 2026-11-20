"""
Resource Management (Lifespan) = Lifecycle
- Startup: Create Book Retriever, Load CLIP, Connect to Qdrant (Only ONCE)
- While Running: Use the same instance not reloading them
- Shutdown: Close the Retriever
"""
