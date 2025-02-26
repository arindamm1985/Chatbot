from fastapi import FastAPI, Header, HTTPException
from pinecone import Pinecone
from openai import OpenAI
import os
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel 

load_dotenv()

# Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "woocommerce-chatbot"

# Initialize Clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("woocommerce-chatbot")

app = FastAPI() 

# Request Models
class EmbedRequest(BaseModel):
    client_id: str
    data: list

class ChatRequest(BaseModel):
    client_id: str
    query: str

# API Root
@app.get("/")
async def root():
    return {"message": "API is live!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)))

# Endpoint to Embed and Upsert content (from client WP plugin)
@app.post("/api/embed")
async def embed_content(req: EmbedRequest):
    if not req.data:
        raise HTTPException(status_code=400, detail="No data provided")

    vectors = []
    for item in req.data:
        content_str = f"Title: {item['title']}, Content: {item['content']}, URL: {item['url']}"

        if 'short_description' in item:
            content_str += f", Short Description: {item['short_description']}"

        if 'categories' in item:
            content_str += f", Categories: {item['categories']}"

        if 'tags' in item:
            content_str += f", Tags: {item['tags']}"

        if 'variations' in item:
            content_str += f", Variations: {item['variations']}"

        if 'shipping' in item:
            content_str += f", Shipping: {item['shipping']}"

        if 'payment' in item:
            content_str += f", Payment Methods: {item['payment']}"

        embedding = openai_client.embeddings.create(
            input=content_str,
            model="text-embedding-ada-002"
        ).data[0].embedding

        vectors.append({
            "id": item["id"],
            "values": embedding,
            "metadata": {
                "title": item["title"],
                "url": item["url"],
                "content": item["content"],
                "featured_image": item.get("featured_image", ""),
                "short_description": item.get("short_description", ""),
                "categories": item.get("categories", ""),
                "tags": item.get("tags", ""),
                "variations": item.get("variations", ""),
                "shipping": item.get("shipping", ""),
                "payment": item.get("payment", "")
            }
        })

    # ✅ Step 2: Store fresh data in Pinecone
    index.upsert(vectors=vectors, namespace=req.client_id)

    return {"success": True, "message": "Namespace cleared and new data embedded successfully"}
    
def rewrite_query(original_query):
    """
    Uses GPT-4 to rewrite the query to make it more contextually rich.
    """
    rewrite_prompt = f"""
    You are an AI assistant optimizing user queries for a vector search database. 
    Rewrite the following query to be more specific and include relevant keywords:
    Each record of this database is either product,post,or page.

    Original query: "{original_query}"
    Improved search query:
    """
    
    chat_response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": rewrite_prompt}],
        temperature=0
    )

    return chat_response.choices[0].message.content.strip()

@app.post("/api/chat")
def chat_with_context(req: ChatRequest, authorization: str = Header(...)):
    # Validate API Key / Authorization here...
    improved_query = rewrite_query(req.query)
    
    query_embedding_resp = openai_client.embeddings.create(
        input=improved_query,
        model="text-embedding-ada-002"
    )

    query_embedding = query_embedding_resp.data[0].embedding

    pinecone_results = index.query(
        vector=query_embedding,
        top_k=3,
        namespace=req.client_id,
        include_metadata=True
    )

    context = "\n---\n".join([
    f"Title: {match['metadata']['title']}\n"
    f"Content: {match['metadata']['content'][:500]}\n"
    f"Short Description: {match['metadata'].get('short_description', 'N/A')}\n"
    f"Categories: {match['metadata'].get('categories', 'N/A')}\n"
    f"Tags: {match['metadata'].get('tags', 'N/A')}\n"
    f"Variations: {match['metadata'].get('variations', 'N/A')}\n"
    f"Shipping Options: {match['metadata'].get('shipping', 'N/A')}\n"
    f"Payment Methods: {match['metadata'].get('payment', 'N/A')}"
    for match in pinecone_results['matches']
])

    prompt = f"""
    You are an AI assistant trained exclusively on our WooCommerce website data.  
    Your goal is to provide direct and helpful responses based on the available data.  

    ### **Response Guidelines:**
    - If the query matches a **product**, return the **response, image, and link**.
    - If the query is general, return a **concise response**.
    - If no relevant data is found, respond with:  
    *"I'm sorry, but I couldn't find relevant information based on our website data."*

    ### **Available Data:**  
    {context}

    ### **Response Format for Products:**
    If the query matches a product, structure the response as follows:  
    ***response***
    <img src="https://example.com/red-sneakers-medium.jpg" width="300px" alt="Red Sneakers" />
    <a href="https://example.com/red-sneakers" target="_blank">Click here</a>

    """

    chat_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    # Corrected response extraction
    answer = chat_response.choices[0].message.content.strip()

    return {
        "success": True,
        "response": answer,
        "client_id": req.client_id,
        "query": req.query,
        "context": context,
        "improved_query": improved_query
    }

@app.get("/")
async def root():
    return {"message": "API running successfully!"}
