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
    content_id: str
    title: str
    content: str

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
async def embed_content(req: EmbedRequest, authorization: str = Header(...)):
    # Validate API Key / Authorization here...

    embedding_resp = openai_client.embeddings.create(
        input=f"{req.title}. {req.content}",
        model="text-embedding-ada-002"
    )

    embedding = embedding_resp.data[0].embedding

    index.upsert(
        vectors=[
            {
                "id": f"{req.client_id}-{req.content_id}",
                "values": embedding,
                "metadata": {
                    "title": req.title,
                    "content": req.content[:1000]  # Limit to reasonable size
                }
            }
        ],
        namespace=req.client_id  # each client has its own namespace
    )

    return {"success": True, "message": "Content indexed successfully."}

@app.post("/api/chat")
def chat_with_context(req: ChatRequest, authorization: str = Header(...)):
    # Validate API Key / Authorization here...

    query_embedding_resp = openai_client.embeddings.create(
        input=req.query,
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
        f"Title: {match['metadata']['title']}\nContent: {match['metadata']['content'][:500]}"
        for match in pinecone_results['matches']
    ])

    prompt = f"""
    You're an assistant trained exclusively on the WooCommerce website data provided.
    Answer only based on the provided context below. If the answer isn't clearly found, politely respond:
    'I'm sorry, but I can't answer that question based on our website data.'

    Website Data:
    {context}

    User Question: {req.query}
    """

    chat_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return {
        "success": True,
        "response": chat_response['choices'][0]['message']['content'].strip()
    }

@app.get("/")
async def root():
    return {"message": "API running successfully!"}
