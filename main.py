from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import openai
import pinecone

app = FastAPI()

openai.api_key = "OPENAI_API_KEY"
pinecone.init(api_key="PINECONE_KEY", environment="PINECONE_ENV")
index = pinecone.Index("your-index-name")

class ChatRequest(BaseModel):
    client_id: str
    query: str

@app.post("/api/chat")
def handle_chat(req: ChatRequest, authorization: str = Header(...)):
    # Validate client_id and authorization token here

    embedding_response = openai.Embedding.create(
        input=req.query,
        model="text-embedding-ada-002"
    )

    embedding = embedding_response['data'][0]['embedding']

    result = index.query(
        vector=embedding,
        top_k=3,
        namespace=req.client_id,  # use namespaces per client
        include_metadata=True
    )

    context = "\n---\n".join([
        f"{m['metadata']['title']}:\n{m['metadata']['content'][:500]}"
        for m in result.matches
    ])

    prompt = f"""
    Answer only based on the provided context:
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
