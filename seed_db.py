"""
seed_db.py
----------
1. Takes the cars CSV (or the hardcoded dataset below).
2. Generates synthetic reviews by mixing review fragments.
3. Inserts everything into a Supabase Postgres table.

Requirements:
    pip install supabase python-dotenv

Environment variables (put in .env):
    SUPABASE_URL=https://xxxx.supabase.co
    SUPABASE_SERVICE_KEY=your-service-role-key
"""

import os
import random
import json
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ---------------------------------------------------------------------------
# Review sentence fragments — randomly assembled into one review per car
# ---------------------------------------------------------------------------
POSITIVE_SENTENCES = [
    "The mileage is excellent for city driving.",
    "Very comfortable seats for long highway drives.",
    "The infotainment system is intuitive and responsive.",
    "Build quality feels solid and well-finished.",
    "Great value for money in this segment.",
    "The suspension handles potholes surprisingly well.",
    "Smooth gear shifts even in stop-and-go traffic.",
    "Boot space is adequate for family weekend trips.",
    "The engine is peppy and responsive at low RPMs.",
    "Safety features like ABS and airbags give confidence.",
    "Service centres are widely available across the country.",
    "Fuel efficiency is better than what brochures claim.",
    "The AC cools the cabin quickly even in peak summer.",
    "Low maintenance costs make ownership stress-free.",
    "Ground clearance is perfect for broken city roads.",
    "Steering feel is light and easy to manoeuvre in traffic.",
    "The rear legroom is surprisingly generous.",
    "Nice exterior styling that turns heads.",
    "Touch controls on the steering wheel are a nice touch.",
    "The headlights are bright and cover a wide area at night.",
]

NEGATIVE_SENTENCES = [
    "Cabin noise at highway speeds could be better insulated.",
    "The touchscreen can lag occasionally.",
    "Rear seat comfort drops on very long journeys.",
    "Could do with more USB charging ports.",
    "The spare tyre is a stepney, which feels dated.",
    "Visibility from the rear could be improved.",
    "Boot lip is a bit high for loading heavy luggage.",
    "Tyre noise intrudes into the cabin on concrete roads.",
    "The instrument cluster design feels slightly dated.",
    "Waiting period can stretch to 4-6 weeks.",
]

NEUTRAL_SENTENCES = [
    "Compared to rivals it holds its own well.",
    "Test drove three alternatives before choosing this one.",
    "Overall a sensible choice for the Indian market.",
    "Would recommend to first-time car buyers.",
    "A solid all-rounder with few compromises.",
    "Perfect for small families with occasional outstation trips.",
    "Exactly what I expected after reading online reviews.",
]


def generate_review(car_name: str) -> str:
    """Stitch together 4-7 random sentences into one review."""
    positives = random.sample(POSITIVE_SENTENCES, k=random.randint(3, 5))
    negatives = random.sample(NEGATIVE_SENTENCES, k=random.randint(1, 2))
    neutrals = random.sample(NEUTRAL_SENTENCES, k=random.randint(1, 2))

    sentences = positives + negatives + neutrals
    random.shuffle(sentences)

    intro = f"I have been driving the {car_name} for about {random.randint(6, 24)} months now. "
    return intro + " ".join(sentences)


def clean_value(value):
    if value == "" or value is None:
        return None
    return value


def cast_record(car):
    return {
        "car_name": car["car_name"],
        "reviews_count": int(car["reviews_count"]) if car["reviews_count"] else None,
        "fuel_type": car["fuel_type"],
        "engine_displacement": int(car["engine_displacement"]) if car["engine_displacement"] else None,
        "no_cylinder": int(car["no_cylinder"]) if car["no_cylinder"] else None,
        "seating_capacity": float(car["seating_capacity"]) if car["seating_capacity"] else None,
        "transmission_type": car["transmission_type"],
        "fuel_tank_capacity": float(car["fuel_tank_capacity"]) if car["fuel_tank_capacity"] else None,
        "body_type": car["body_type"],
        "rating": float(car["rating"]) if car["rating"] else None,
        "starting_price": int(car["starting_price"]) if car["starting_price"] else None,
        "ending_price": int(car["ending_price"]) if car["ending_price"] else None,
        "max_torque_nm": float(car["max_torque_nm"]) if car["max_torque_nm"] else None,
        "max_torque_rpm": int(car["max_torque_rpm"]) if car["max_torque_rpm"] else None,
        "max_power_bhp": float(car["max_power_bhp"]) if car["max_power_bhp"] else None,
        "max_power_rpm": int(car["max_power_rpm"]) if car["max_power_rpm"] else None,
    }

import csv

with open("cars.csv") as f:
    CARS = list(csv.DictReader(f))
    # Cast numeric fields as needed

# ---------------------------------------------------------------------------
# SQL to create the table (run once in Supabase SQL editor or via RPC)
# ---------------------------------------------------------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cars (
    id                  BIGSERIAL PRIMARY KEY,
    car_name            TEXT NOT NULL,
    reviews_count       INTEGER,
    fuel_type           TEXT,
    engine_displacement INTEGER,
    no_cylinder         INTEGER,
    seating_capacity    NUMERIC,
    transmission_type   TEXT,
    fuel_tank_capacity  NUMERIC,
    body_type           TEXT,
    rating              NUMERIC,
    starting_price      BIGINT,
    ending_price        BIGINT,
    max_torque_nm       NUMERIC,
    max_torque_rpm      INTEGER,
    max_power_bhp       NUMERIC,
    max_power_rpm       INTEGER,
    reviews             TEXT
);
"""


def seed():
    print("=== CarDekho DB Seeder ===\n")

    # NOTE: Supabase Python client doesn't execute raw DDL.
    # Print the SQL so you can run it in the Supabase SQL editor first.
    print("Run this SQL in Supabase SQL editor first:\n")
    print(CREATE_TABLE_SQL)
    print("\nThen press Enter to continue with data insertion...")
    input()

    records = []
    for car in CARS:
        review = generate_review(car["car_name"])
        # record = {**car, "reviews": review}
        cleaned = cast_record(car)
        record = {**cleaned, "reviews": review}
        records.append(record)
        print(f"  Generated review for: {car['car_name']}")

    print(f"\nInserting {len(records)} cars into Supabase...")
    result = supabase.table("cars").insert(records).execute()

    if result.data:
        print(f"\n✓ Successfully inserted {len(result.data)} rows.")
    else:
        print(f"\n✗ Insertion failed. Response: {result}")

    print("\n=== Done! ===")


if __name__ == "__main__":
    seed()