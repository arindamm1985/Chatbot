from fastapi import FastAPI, Header, HTTPException
from pinecone import Pinecone
from openai import OpenAI
import os
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel 
import requests
from bs4 import BeautifulSoup
import json
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
class ExtractRequest(BaseModel):
    url: str
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
def extract_sections(url):
    # Fetch the page content
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/90.0.4430.93 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    response = requests.get(url, headers=headers)
    
    response.raise_for_status()  # Raise an error for bad responses
    html = response.text

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove header, footer, and logo (if present)
    for tag in soup.find_all(["header", "footer"]):
        tag.decompose()
    logo = soup.find('img', id='logo')
    if logo:
        logo.decompose()
    
    # Use the <main> tag if available; otherwise, use the body.
    main_content = soup.find('main') or soup.body

    sections = []
    # Iterate over direct child div elements of the main content
    for div in main_content.find_all('div', recursive=False):
        section_data = {}
        # Attempt to extract a section title by looking for the first heading element (h1-h6)
        title_tag = div.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        section_data['title'] = title_tag.get_text(strip=True) if title_tag else ""
        # Extract immediate paragraph text as the section's content
        paragraphs = div.find_all('p', recursive=False)
        section_data['content'] = " ".join(p.get_text(strip=True) for p in paragraphs)
        
        # Extract nested child sections: look for direct child divs that contain a heading or paragraph
        child_sections = []
        for child in div.find_all('div', recursive=False):
            if child.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']) or child.find('p'):
                child_data = {}
                child_title = child.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                child_data['title'] = child_title.get_text(strip=True) if child_title else ""
                child_paragraphs = child.find_all('p', recursive=False)
                child_data['content'] = " ".join(p.get_text(strip=True) for p in child_paragraphs)
                child_sections.append(child_data)
        section_data['child_sections'] = child_sections
        
        # Only include the section if there's some content (title or text)
        if section_data['title'] or section_data['content'] or child_sections:
            sections.append(section_data)
    
    # Return the structured data as JSON
    return {"sections": sections}

@app.post('/api/extract')
async def api_extract(req: ExtractRequest):
    url = req.url
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided.")
    
    return extract_sections(url)
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

    For each user query, produce a two-part answer:
    1. A concise one‑line plain text answer that directly addresses the query.
    2. A complete valid HTML version of the answer below the plain text answer. Do not include any labels like "Plain-Text Answer:" or "HTML Output:"—simply output the plain text answer on the first line, then a blank line, then the HTML code.

    Below are examples of how to handle different queries:

    Example 1:
    User Query: "Hi"
    Desired Output:
    Hello, how can I help?

    Example 2:
    User Query: "Do you have stickers?"
    Desired Output:
    Yes, we have a variety of stickers available in different categories and variations.
    <ul>
      <li>
        <img src="https://liveprojectscare.com/woocomercechagpt/sticker1.jpg" alt="Sticker 1" style="width:50px;">
        <strong>Sticker 1</strong> - $4.00 
        <a href="https://liveprojectscare.com/woocomercechagpt/sticker1">View Details</a>
      </li>
      <li>
        <img src="https://liveprojectscare.com/woocomercechagpt/sticker2.jpg" alt="Sticker 2" style="width:50px;">
        <strong>Sticker 2</strong> - $5.00 
        <a href="https://liveprojectscare.com/woocomercechagpt/sticker2">View Details</a>
      </li>
    </ul>
    <p>Please let me know which one you like.</p>

    Example 3:
    User Query: "What are your working hours?"
    Desired Output:
    Our working hours are between 6am and 9pm.

    Example 4:
    User Query: "Which sticker is better, holographic or white vinyl?"
    Desired Output:
    Both are excellent, but they offer different benefits.
    <table border="1" cellspacing="0" cellpadding="5">
      <tr>
        <th>Feature</th>
        <th>Holographic Stickers</th>
        <th>White Vinyl Stickers</th>
      </tr>
      <tr>
        <td>Durability</td>
        <td>High</td>
        <td>Medium</td>
      </tr>
      <tr>
        <td>Finish</td>
        <td>Shiny</td>
        <td>Matte</td>
      </tr>
      <tr>
        <td>Price</td>
        <td>$4.50</td>
        <td>$4.00</td>
      </tr>
    </table>
    <p>Which one would you like to know more about?</p>

    Now, using the context data from Pinecone:
    {context}

    And the user's optimized query:
    {improved_query}

    Please generate your final answer as follows:
    - On the first line, output the plain text answer.
    - Leave a blank line.
    - Then output the complete valid HTML response, without any escaped characters (use standard quotes).

    Make sure the HTML formatting fits the query type, and the overall response is concise, friendly, and sales-focused.
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
