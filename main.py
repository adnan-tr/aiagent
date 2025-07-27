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

# HuggingFace embedding model
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_API_KEY = "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"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# Precompute product embeddings
def get_embedding(texts):
    response = httpx.post(HF_API_URL, headers=HEADERS, json={"inputs": texts})
    response.raise_for_status()
    return np.array(response.json())

embeddings = get_embedding(df['text'].tolist())

class MatchRequest(BaseModel):
    query: str

@app.post("/match")
async def match_product(req: MatchRequest):
    query_emb = get_embedding([req.query])
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:3]
    results = df.iloc[top_indices][['Product Code', 'Product Name', 'Specs']].to_dict(orient="records")
    return {"matches": results}
