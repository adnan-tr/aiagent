from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

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

# Load embedding model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Precompute embeddings
embeddings = model.encode(df['text'].tolist(), convert_to_tensor=True)

class MatchRequest(BaseModel):
    query: str

@app.post("/match")
async def match_product(req: MatchRequest):
    query_emb = model.encode([req.query], convert_to_tensor=True)
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:3]
    results = df.iloc[top_indices][['Product Code', 'Product Name', 'Specs']].to_dict(orient="records")
    return {"matches": results}
