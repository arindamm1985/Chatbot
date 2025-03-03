from fastapi import FastAPI, Header, HTTPException
from pinecone import Pinecone
from openai import OpenAI
import os
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel 
import requests
from bs4 import BeautifulSoup
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain.tools import Tool
from langchain-openai import OpenAI
from langchain_pinecone import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
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
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

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
    
memory_store = {}

### Function to fetch context from Pinecone using a given namespace (client_id)
def fetch_context(query: str, namespace: str) -> str:
    # Create the embedding for the query using LangChain's embeddings helper
    query_embedding = embeddings.embed_query(query)
    
    pinecone_results = index.query(
        vector=query_embedding,
        top_k=3,
        namespace=namespace,
        include_metadata=True
    )
    
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
    
    return context if context else "No relevant data found."

# In get_agent(), we build our tool list using a local version of fetch_context that passes client_id as namespace.
def get_agent(client_id: str):
    # Create or get memory for this client
    if client_id not in memory_store:
        memory_store[client_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory = memory_store[client_id]

    llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    
    # Define a tool that fetches context using the client_id as namespace
    fetch_context_tool = Tool(
        name="Fetch Relevant Context",
        func=lambda query: fetch_context(query, client_id),
        description="Retrieves metadata and content from the Pinecone vector database using the client_id as the namespace."
    )
    
    # (Additional tools such as product search or order tracking could be added here if needed.)
    tools = [fetch_context_tool]
    
    # Initialize the agent with the tools and memory
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )
    
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory
    )
    
    return executor, memory

# API Request Model
class ChatRequest(BaseModel):
    query: str
    client_id: str

@app.post("/api/chat")
def chat_with_context(req: ChatRequest, authorization: str = Header(...)):
    try:
        # Get the agent executor and memory for this client_id (which is also used as namespace)
        executor, memory = get_agent(req.client_id)
        
        # (Chat history is automatically managed via the memory instance.)
        
        # Execute the agent with the user query; the agent will invoke the fetch_context tool internally.
        response = executor.run(req.query)
        
        # Store the query/response in memory
        memory.save_context({"input": req.query}, {"output": response})
        
        return {
            "success": True,
            "response": response,
            "client_id": req.client_id,
            "query": req.query,
            "chat_history": memory.load_memory_variables({}).get("chat_history", "No previous history.")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "API running successfully!"}
