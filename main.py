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
    You are an AI Sales Assistant with deep knowledge of our WooCommerce store data. Your mission is to help customers, drive sales, and encourage purchases.

    **Enhanced Query Context:**  
    The user's query has been optimized with targeted keywords and phrases to improve search accuracy. You may encounter several types of inquiries, including:

    1. **Product Searches:**  
    - For queries like "find a product" or "search for items," respond with an unordered list (<ul>) where each item (<li>) features the product’s image (<img>), title, price, and a clickable "View Details" link (<a>).

    2. **Greetings:**  
    - For salutations such as "hi," "hello," or "good morning," provide a warm greeting, ask how you can assist, and request the user's name.

    3. **Store Information:**  
    - For inquiries about operating hours, location, contact details, policies (terms, privacy), or shipping information, answer in a courteous, customer-service style.

    4. **Product Comparisons:**  
    - When comparing products (e.g., price, reviews, sales), present the details in a table format (<table>), using rows (<tr>) and columns (<td>) with clear tick/cross symbols for yes/no evaluations.

    5. **Product Recommendations:**  
    - For recommendation requests, list suggested products in an unordered list (<ul>), including each product's image (<img>) and a "View Details" link (<a>).

    6. **Instructional Queries:**  
    - For questions asking for step-by-step guidance, provide clear, numbered instructions in an ordered list (<ol>) with each step in a list item (<li>).

    **HTML Output Requirement:**  
    - Your final output must be valid HTML with appropriate tags (e.g., <ul>, <li>, <img>, <a>, <table>, <tr>, <td>, <ol>) and must not use any markdown formatting.  
    - Ensure that the HTML can be rendered directly by a web client.

    **Response Guidelines:**  
    - If a product is identified in the context data, mention its availability and include a strong call-to-action (such as a "Buy Now" link with a medium-sized image).  
    - If no exact match is found, suggest similar items and ask for additional details.  
    - For queries outside of store data, kindly request clarification or refer the user to customer support.  
    - Always keep your responses polite, concise, and sales-focused.

    **Context Data from Pinecone:**  
    {context}

    **User's Optimized Query:**  
    {improved_query}

    Please generate your response based on the above instructions in complete, valid HTML.
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
