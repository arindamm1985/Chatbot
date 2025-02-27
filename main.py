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

class EmptyNamespaceRequest(BaseModel):
    client_id: str

@app.post("/api/empty-namespace")
def empty_pinecone_namespace(request: EmptyNamespaceRequest):
    try:
        namespace = request.client_id

        # ✅ Delete all vectors from the specified namespace
        index.delete(delete_all=True, namespace=namespace)

        return {"success": True, "message": f"Namespace '{namespace}' has been emptied."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
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
    Optimizes the user's query for our product search database.
    It converts the query into specific keywords and relevant phrases to target product titles,
    descriptions, categories, tags, variations, shipping, and payment details.
    It also considers whether the query is a greeting, store information request, product search,
    comparison, recommendation, or an inquiry for instructions.
    """
    rewrite_prompt = f"""
    You are an AI assistant specialized in optimizing user queries for a product search database.
    Rewrite the following query to include specific keywords and relevant phrases that target product titles, descriptions,
    categories, tags, variations, shipping, and payment details. Consider if the query is a greeting, a store info request,
    a product search, a comparison, a recommendation, or a request for instructions, and include the appropriate keywords.
    Original query: "{original_query}"
    Optimized query:
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

    # Step 1: Rewrite the user's query to include specific keywords and phrases.
    improved_query = rewrite_query(req.query)
    
    # Step 2: Get an embedding for the optimized query.
    query_embedding_resp = openai_client.embeddings.create(
        input=improved_query,
        model="text-embedding-ada-002"
    )
    query_embedding = query_embedding_resp.data[0].embedding

    # Step 3: Query Pinecone using the optimized embedding.
    pinecone_results = index.query(
        vector=query_embedding,
        top_k=3,
        namespace=req.client_id,
        include_metadata=True
    )

    # Step 4: Build context string from the returned matches.
    context = "\n---\n".join([
        f"Title: {match['metadata']['title']}\n"
        f"Content: {match['metadata']['content'][:500]}\n"
        f"Url: {match['metadata']['url'][:500]}\n"
        f"Image: {match['metadata']['featured_image'][:500]}\n"
        f"Short Description: {match['metadata'].get('short_description', 'N/A')}\n"
        f"Categories: {match['metadata'].get('categories', 'N/A')}\n"
        f"Tags: {match['metadata'].get('tags', 'N/A')}\n"
        f"Variations: {match['metadata'].get('variations', 'N/A')}\n"
        f"Shipping Options: {match['metadata'].get('shipping', 'N/A')}\n"
        f"Payment Methods: {match['metadata'].get('payment', 'N/A')}"
        for match in pinecone_results['matches']
    ])

    # Step 5: Construct the final prompt for generating the sales response.
    prompt = f"""
    You are an AI-powered Sales Assistant trained on our WooCommerce store data.  
    Your primary goal is to assist potential customers, drive sales, and encourage purchases.

    ### User Query Optimization:
    The user's query has been optimized to include relevant keywords and phrases for matching our database. The possible query types include:
    a) **Product Searches:** (e.g., "find a product", "search for items")  
    - Respond with an unordered list displaying each product's feature image, title, and a clickable "View Details" link.
    b) **Greetings:** (e.g., "hi", "hello", "good morning")  
    - Reply with a friendly greeting, ask how you can help, and ask for the user's name.
    c) **Store Information:** (e.g., inquiries about hours, location, contact details, terms, privacy, shipping)  
    - Respond like a courteous customer care executive.
    d) **Product Comparisons:** (e.g., comparing price, reviews, sales)  
    - Use a table format with columns for features and tick/cross symbols for yes/no.
    e) **Product Recommendations:**  
    - Provide suggestions in an unordered list with a product image and "View Details" link.
    f) **Instructional Queries:** (e.g., asking for steps to complete a task)  
    - Provide clear, numbered step-by-step instructions.

    ### Response Guidelines:
    - If a product is found in the context, include a brief note about its availability along with a strong call to action (e.g., a "Buy Now" link with a medium-sized image).
    - If no direct match is found, suggest similar items and ask for more details.
    - For queries not directly related to store data, kindly ask for clarification or direct the user to customer support.
    - Keep your responses polite, concise, and sales-focused.

    ### Context Data from Pinecone:
    {context}

    ### User's Original Query (Optimized):
    {improved_query}

    Please generate a response based on the above instructions and context.
    """

    chat_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

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
