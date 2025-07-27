from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import httpx
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load and prepare product data
df = pd.read_csv("product_data.csv")
df['Product Code'] = df['MATERIAL NO']
df['Product Name'] = df['MATERIAL DESCRIPTION']
df['Specs'] = (
    "Net weight: " + df['NET'].astype(str) + " kg, " +
    "Gross weight: " + df['GROSS'].astype(str) + " kg, " +
    "Dimensions: " + df['DIMENSIONS (mm)'].astype(str)
)
df['text'] = df['Product Name'] + " " + df['Specs']

# OpenRouter config
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://yourdomain.com",
    "Content-Type": "application/json"
}

def get_embedding(text: str) -> np.ndarray:
    prompt = f"Convert this to a 10-dimensional numerical embedding only, no explanation:\n\n{text}"
    body = {
        "model": "mistralai/mixtral-8x7b",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = httpx.post(OPENROUTER_URL, headers=HEADERS, json=body, timeout=30.0)
    response.raise_for_status()
    result = response.json()
    vector_text = result['choices'][0]['message']['content']
    vector = [float(x.strip()) for x in vector_text.strip("[]").split(",")]
    return np.array(vector)

# Precompute product embeddings
product_embeddings = np.array([get_embedding(text) for text in df['text']])

class MatchRequest(BaseModel):
    query: str

@app.post("/match")
async def match_product(req: MatchRequest):
    query_vec = get_embedding(req.query)
    scores = cosine_similarity([query_vec], product_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:3]
    results = df.iloc[top_indices][['Product Code', 'Product Name', 'Specs']].to_dict(orient="records")
    return {"matches": results}
