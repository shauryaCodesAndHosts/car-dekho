"""
embed_pinecone.py
-----------------
Fetches all cars from Supabase, embeds the review text using Cohere,
and upserts vectors into Pinecone for semantic similarity search.

Requirements:
    pip install supabase cohere pinecone-client python-dotenv

Environment variables (in .env):
    SUPABASE_URL=...
    SUPABASE_SERVICE_KEY=...
    COHERE_API_KEY=...
    PINECONE_API_KEY=...
    PINECONE_INDEX_NAME=cardekho-reviews   (create this index in Pinecone dashboard)
    PINECONE_ENVIRONMENT=...               (e.g. us-east-1)
"""

import os
import time
from dotenv import load_dotenv
from supabase import create_client, Client
import cohere
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "cardekho-reviews")

# Cohere embed-english-v3.0 outputs 1024-dimensional vectors
EMBEDDING_DIM = 1024

def safe_float(value, default=0.0):
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def get_all_cars(supabase: Client):
    result = supabase.table("cars").select("*").execute()
    return result.data


def embed_texts(co: cohere.Client, texts: list[str]) -> list[list[float]]:
    """
    Embed texts in batches of 96 (Cohere limit).
    input_type='search_document' is correct for indexing.
    """
    all_embeddings = []
    batch_size = 96

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = co.embed(
            texts=batch,
            model="embed-english-v3.0",
            input_type="search_document",
        )
        all_embeddings.extend(response.embeddings)
        print(f"  Embedded batch {i // batch_size + 1} ({len(batch)} texts)")
        time.sleep(0.5)  # rate-limit guard

    return all_embeddings


def ensure_index(pc: Pinecone):
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' ...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            print("  Waiting for index to be ready...")
            time.sleep(5)
        print("  Index ready.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")


def upsert_to_pinecone(index, cars: list[dict], embeddings: list[list[float]]):
    """
    Each vector ID = str(car['id'])
    Metadata stores filterable fields so we can do metadata-filtered ANN search.
    """
    vectors = []
    for car, embedding in zip(cars, embeddings):
        metadata = {
            "car_name": car.get("car_name"),
            "body_type": car.get("body_type"),
            "fuel_type": car.get("fuel_type"),
            "transmission_type": car.get("transmission_type"),
            "seating_capacity": safe_float(car.get("seating_capacity")),
            "starting_price": safe_int(car.get("starting_price")),
            "ending_price": safe_int(car.get("ending_price")),
            "rating": safe_float(car.get("rating")),
            "reviews_count": safe_int(car.get("reviews_count")),
        }
        vectors.append(
            {
                "id": str(car["id"]),
                "values": embedding,
                "metadata": metadata,
            }
        )

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"  Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")


def main():
    print("=== CarDekho Pinecone Embedder ===\n")

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    co = cohere.Client(COHERE_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    print("Fetching cars from Supabase...")
    cars = get_all_cars(supabase)
    print(f"Found {len(cars)} cars.\n")

    texts = [car["reviews"] for car in cars]

    print("Embedding reviews with Cohere...")
    embeddings = embed_texts(co, texts)
    print(f"\nGenerated {len(embeddings)} embeddings.\n")

    ensure_index(pc)
    index = pc.Index(PINECONE_INDEX_NAME)

    print("\nUpserting to Pinecone...")
    upsert_to_pinecone(index, cars, embeddings)

    stats = index.describe_index_stats()
    print(f"\n✓ Pinecone index stats: {stats}")
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()