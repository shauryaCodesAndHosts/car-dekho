"""
main.py — CarDekho AI Advisor Backend
--------------------------------------
Pipeline per request:
  1. Groq LLM → extract structured filters from natural language query
  2. Supabase → filter cars by hard rules (up to 20 results)
  3. Cohere → embed user query
  4. Pinecone → similarity rerank against review embeddings
  5. Top 10 cars → Groq LLM → natural language recommendation

Requirements:
    pip install fastapi uvicorn supabase cohere pinecone-client groq python-dotenv pydantic

Run:
    uvicorn main:app --reload --port 8000
"""

import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import cohere
from pinecone import Pinecone
from supabase import create_client, Client
from groq import Groq

load_dotenv()

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
supabase: Client = create_client(
    os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"]
)
co = cohere.Client(os.environ["COHERE_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pinecone_index = pc.Index(os.environ.get("PINECONE_INDEX_NAME", "cardekho-reviews"))
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

GROQ_MODEL = "llama-3.1-8b-instant"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="CarDekho AI Advisor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str


class CarResult(BaseModel):
    id: int
    car_name: str
    body_type: str
    fuel_type: str
    transmission_type: str
    seating_capacity: float
    starting_price: int
    ending_price: int
    rating: float
    reviews_count: int
    max_power_bhp: float
    fuel_tank_capacity: float
    reviews: str


class AdvisorResponse(BaseModel):
    query: str
    extracted_filters: dict
    cars: list[CarResult]
    recommendation: str


# ---------------------------------------------------------------------------
# Step 1: Extract filters from natural language query
# ---------------------------------------------------------------------------
FILTER_EXTRACTION_PROMPT = """
You are a car filter extraction engine. Given a user query, extract structured 
search parameters. Return ONLY valid JSON with these optional fields:

{{
  "fuel_type": "Petrol" | "Diesel" | "Electric" | null,
  "body_type": "Hatchback" | "Sedan" | "SUV" | "MPV" | null,
  "transmission_type": "Automatic" | "Manual" | null,
  "min_seating_capacity": number | null,
  "max_price": number | null,
  "min_price": number | null,
  "min_rating": number | null,
  "min_power_bhp": number | null,
  "max_engine_displacement": number | null,
  "min_fuel_tank": number | null
}}

Rules:
- Prices are in Indian Rupees.
- STRICTLY follow:
    1 lakh = 100000
    10 lakh = 1000000
    15 lakh = 1500000
    1 crore = 10000000
- Never multiply lakh by 1,000,000.
- If the user says "budget", that maps to max_price.
- Return null for any field not mentioned.
- Return ONLY the JSON object. No explanation, no markdown.

User query: {query}
"""


def extract_filters(query: str) -> dict:
    prompt = FILTER_EXTRACTION_PROMPT.format(query=query)
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if Groq wraps it
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        filters = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return empty filters instead of crashing
        filters = {}

    return filters


# ---------------------------------------------------------------------------
# Step 2: Query Supabase with extracted filters
# ---------------------------------------------------------------------------
def query_supabase(filters: dict, limit: int = 20) -> list[dict]:
    q = supabase.table("cars").select("*")

    if filters.get("fuel_type"):
        q = q.eq("fuel_type", filters["fuel_type"])

    if filters.get("body_type"):
        q = q.eq("body_type", filters["body_type"])

    if filters.get("transmission_type"):
        q = q.eq("transmission_type", filters["transmission_type"])

    if filters.get("min_seating_capacity") is not None:
        q = q.gte("seating_capacity", float(filters["min_seating_capacity"]))

    if filters.get("max_price") is not None:
        q = q.lte("starting_price", filters["max_price"])

    if filters.get("min_price") is not None:
        q = q.gte("starting_price", filters["min_price"])

    if filters.get("min_rating") is not None:
        q = q.gte("rating", filters["min_rating"])

    if filters.get("min_power_bhp") is not None:
        q = q.gte("max_power_bhp", filters["min_power_bhp"])

    if filters.get("min_fuel_tank") is not None:
        q = q.gte("fuel_tank_capacity", filters["min_fuel_tank"])

    result = q.limit(limit).execute()
    return result.data or []


# ---------------------------------------------------------------------------
# Step 3: Embed user query with Cohere
# ---------------------------------------------------------------------------
def embed_query(query: str) -> list[float]:
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",  # 'search_query' for query-side embedding
    )
    return response.embeddings[0]


# ---------------------------------------------------------------------------
# Step 4: Rerank candidates using Pinecone similarity
# ---------------------------------------------------------------------------
def rerank_with_pinecone(
    candidates: list[dict], query_embedding: list[float], top_k: int = 10
) -> list[dict]:
    if not candidates:
        return []

    candidate_ids = [str(c["id"]) for c in candidates]

    # Fetch existing vectors from Pinecone for our candidates
    fetch_result = pinecone_index.fetch(ids=candidate_ids)
    vectors_map = fetch_result.vectors  # dict: id -> Vector object

    # Compute cosine similarity manually for each candidate
    def cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x**2 for x in a) ** 0.5
        norm_b = sum(x**2 for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    scored = []
    for car in candidates:
        car_id = str(car["id"])
        if car_id in vectors_map:
            vec = vectors_map[car_id].values
            score = cosine_sim(query_embedding, vec)
        else:
            score = 0.0
        scored.append((score, car))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [car for _, car in scored[:top_k]]


# ---------------------------------------------------------------------------
# Step 5: Generate recommendation with Groq
# ---------------------------------------------------------------------------
RECOMMENDATION_PROMPT = """
You are a knowledgeable and friendly car buying advisor for CarDekho, India's 
leading car research platform.

A buyer has asked: "{query}"

Based on their requirements, here are the top matching cars:

{car_summaries}

Write a warm, helpful recommendation in 200-300 words. 
- Start by acknowledging what the buyer is looking for.
- Highlight the top 2-3 best matches and WHY they suit the buyer specifically.
- Mention key specs that align with the buyer's needs.
- End with a confident shortlist suggestion.
- Use simple language. No bullet points — write in flowing paragraphs.
- Be specific, not generic. Reference actual car names and prices.
"""


def format_car_for_prompt(car: dict) -> str:
    price_str = f"₹{car['starting_price']:,} – ₹{car['ending_price']:,}"
    return (
        f"• {car['car_name']} ({car['body_type']}, {car['fuel_type']}, "
        f"{car['transmission_type']}) | {price_str} | "
        f"Rating: {car['rating']}/5 | Power: {car['max_power_bhp']} bhp | "
        f"Seats: {int(car['seating_capacity'])} | "
        f"Review excerpt: \"{car['reviews'][:200]}...\""
    )


def generate_recommendation(query: str, cars: list[dict]) -> str:
    car_summaries = "\n".join(format_car_for_prompt(c) for c in cars)
    prompt = RECOMMENDATION_PROMPT.format(query=query, car_summaries=car_summaries)

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
@app.post("/recommend", response_model=AdvisorResponse)
async def recommend(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Step 1: Extract filters
    filters = extract_filters(request.query)

    # Step 2: Fetch from Supabase
    candidates = query_supabase(filters, limit=20)

    if not candidates:
        # Fallback: return top-rated cars if no filters matched
        fallback = supabase.table("cars").select("*").order("rating", desc=True).limit(10).execute()
        candidates = fallback.data or []

    # Step 3: Embed query
    query_embedding = embed_query(request.query)

    # Step 4: Rerank
    top_cars = rerank_with_pinecone(candidates, query_embedding, top_k=10)

    # If Pinecone fetch failed (empty index), just take top 10 by rating
    if not top_cars:
        top_cars = sorted(candidates, key=lambda x: x.get("rating", 0), reverse=True)[:10]

    # Step 5: Generate recommendation
    recommendation = generate_recommendation(request.query, top_cars)

    return AdvisorResponse(
        query=request.query,
        extracted_filters=filters,
        cars=[CarResult(**c) for c in top_cars],
        recommendation=recommendation,
    )


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "index2.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok", "service": "CarDekho AI Advisor"}


GUIDE_STEPS = [
    {
        "key": "budget",
        "question": "What's your budget?",
        "placeholder": "e.g. under 12 lakhs",
    },
    {
        "key": "body_type",
        "question": "What kind of car are you looking for?",
        "placeholder": "SUV, Hatchback, Sedan, MPV — or say any",
        "options": ["SUV", "Hatchback", "Sedan", "MPV", "Any"],
    },
    {
        "key": "fuel",
        "question": "Petrol, Diesel, or Electric?",
        "placeholder": "e.g. Petrol",
        "options": ["Petrol", "Diesel", "Electric", "No preference"],
    },
    {
        "key": "transmission",
        "question": "Automatic or Manual?",
        "placeholder": "e.g. Automatic",
        "options": ["Automatic", "Manual", "No preference"],
    },
    {
        "key": "seats",
        "question": "How many seats do you need?",
        "placeholder": "e.g. 5 or 7",
        "options": ["4", "5", "7", "Any"],
    },
    {
        "key": "use_case",
        "question": "Last one — how will you mainly use it?",
        "placeholder": "e.g. daily city commute, highway trips, first car for family",
    },
]


class GuideRequest(BaseModel):
    step: int  # next step to serve (0 = first question)
    answers: dict  # answers collected so far { key: value }


class GuideResponse(BaseModel):
    step: int
    question: str
    placeholder: str
    options: list[str]
    done: bool
    assembled_query: str


def assemble_query(answers: dict) -> str:
    parts = []
    budget = answers.get("budget", "").strip()
    if budget and budget.lower() not in ("skip", "any", ""):
        parts.append(f"budget {budget}")
    body = answers.get("body_type", "").strip()
    if body and body.lower() not in ("any", "no preference", ""):
        parts.append(body)
    fuel = answers.get("fuel", "").strip()
    if fuel and fuel.lower() not in ("no preference", ""):
        parts.append(fuel)
    transmission = answers.get("transmission", "").strip()
    if transmission and transmission.lower() not in ("no preference", ""):
        parts.append(transmission)
    seats = answers.get("seats", "").strip()
    if seats and seats.lower() not in ("any", ""):
        parts.append(f"{seats} seats")
    use_case = answers.get("use_case", "").strip()
    if use_case and use_case.lower() not in ("skip", ""):
        parts.append(f"for {use_case}")
    if not parts:
        return "recommend me a good car"
    return "I'm looking for a " + ", ".join(parts)


@app.post("/guide", response_model=GuideResponse)
def guide(req: GuideRequest):
    if req.step >= len(GUIDE_STEPS):
        return GuideResponse(
            step=req.step, question="", placeholder="", options=[],
            done=True, assembled_query=assemble_query(req.answers),
        )
    s = GUIDE_STEPS[req.step]
    return GuideResponse(
        step=req.step,
        question=s["question"],
        placeholder=s.get("placeholder", ""),
        options=s.get("options", []),
        done=False,
        assembled_query="",
    )

