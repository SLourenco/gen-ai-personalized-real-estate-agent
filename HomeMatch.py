#A starter file for the HomeMatch application if you want to build your solution in a Python program instead of a notebook.
import json
import os
from typing import Union

import numpy as np
import lancedb
from dotenv import load_dotenv
from lancedb.pydantic import Vector, LanceModel
from langchain_community.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

print(os.environ["OPENAI_API_KEY"])
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model=model_name)

transformer_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

class RealEstateListing(LanceModel):
    vector: Vector(384)
    neighborhood: str
    price: float
    bedrooms: int
    bathrooms: int
    houseSize: str
    description: str
    neighborhoodDescription: str

connection = lancedb.connect("database")
connection.drop_table("listings", ignore_missing=True)
table = connection.create_table("listings", schema=RealEstateListing, exist_ok=True)


def generate_real_estate_listings():
    try:
        response = llm.invoke(
            """
                Generate at least 10 real estate ads in the following JSON format.

                Each ad should use this schema:
                {
                  "Neighborhood": "string",
                  "Price": "float",
                  "Bedrooms": "number",
                  "Bathrooms": "int",
                  "House Size": "string",
                  "Description": "string",
                  "Neighborhood Description": "string"
                }
                
                Return only a **JSON array** of 10 ads, without any commentary or numbering. Example ad:
                
                {
                  "Neighborhood": "Green Oaks",
                  "Price": 800000,
                  "Bedrooms": 3,
                  "Bathrooms": 2,
                  "House Size": "2000 sqft",
                  "Description": "Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.",
                  "Neighborhood Description": "Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze."
                }
                
                Only return valid JSON.
            """
        )

        return json.loads(response.content)
    except Exception as e:
        return f"An error occurred generating the listings: {e}"

def generate_embeddings(phrase: str):
    e = transformer_model.encode(str(phrase))
    return e

def create_and_save_listings():
    if os.environ["GENERATE_LISTINGS"] == "True":
        # Step 2: Generating Real Estate Listings
        listings = generate_real_estate_listings()
        print("\nListings:")
        print(listings)

        os.remove("listings.json")
        with open("listings.json", "w") as f:
            json.dump(listings, f, indent=4)
    else:
        with open("listings.json") as f:
            listings = json.load(f)

    # Step 3: Storing Listings in a Vector Database
    data = []
    for listing in listings:
        data.append(RealEstateListing(
            vector=generate_embeddings(listing),  # Placeholder for vector
            neighborhood=listing["Neighborhood"],
            price=listing["Price"],
            bedrooms=listing["Bedrooms"],
            bathrooms=listing["Bathrooms"],
            houseSize=listing["House Size"],
            description=listing["Description"],
            neighborhoodDescription=listing["Neighborhood Description"]
        ))

    table.add(data)

create_and_save_listings()

# Step 4: Building the User Preference Interface
def interpret_user_preferences(pref: str):
    try:
        prompt = """
                Based on the user preferences below, generate a real estate listing in json format. Use this schema:
                {
        "Neighborhood": "string",
                  "Price": "float",
                  "Bedrooms": "number",
                  "Bathrooms": "int",
                  "House Size": "string",
                  "Description": "string",
                  "Neighborhood Description": "string"
                }
                
                Example ad:
                {
        "Neighborhood": "Green Oaks",
                  "Price": 800000,
                  "Bedrooms": 3,
                  "Bathrooms": 2,
                  "House Size": "2000 sqft",
                  "Description": "Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.",
                  "Neighborhood Description": "Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze."
                }
                
                Only return valid JSON.
                
                User Preferences:
            """ + pref
        response = llm.invoke(prompt)
        return json.loads(response.content)
    except Exception as e:
        return f"An error occurred interpreting user preferences: {e}"


with open('user_preferences.json') as f:
    user_preferences = json.load(f)

print("\nUser Preferences:")
print(user_preferences)
ideal_listing = interpret_user_preferences(str(user_preferences))

# Step 5: Searching Based on Preferences
user_preferences_vector = generate_embeddings(str(user_preferences))
results = (table.search(user_preferences_vector)
           .where(f"bedrooms = {ideal_listing['Bedrooms']}", prefilter=True)
           .where(f"bathrooms = {ideal_listing['Bathrooms']}", prefilter=True)
           .limit(3).to_pydantic(RealEstateListing))

print("\nRaw Results:")
print(results)

# Step 6: Personalizing Listing Descriptions
def augment_listings(listings: list[RealEstateListing], user_listing: RealEstateListing):
    try:
        prompt = """
                You are a real estate copywriter. A user has shared an "ideal" listing that reflects their personal preferences.
                Your task is to take a list of existing real estate listings and rewrite their Description and Neighborhood Description to emphasize aspects that align with the user's preferences.
                
                You MUST:
                - Maintain factual accuracy — do not add, remove, or invent any features.
                - Reorder, rephrase, or emphasize features already present to better match the tone and focus of the ideal listing.
                - Use a tone similar to the ideal listing: vivid, lifestyle-focused, and emotionally engaging.
                
                You MUST NOT:
                - Change prices, bedroom/bathroom counts, or invent features not already in the listing.

                ### Example Ideal Listing:
                """ + str(user_listing) + """
                
                ### Listings to Rewrite: """ + str(listings) + """
                
                Only return the list of rewritten JSON objects, in the same format.
                Only return valid JSON.
            """

        response = llm.invoke(prompt)
        return json.loads(response.content)
    except Exception as e:
        return f"An error occurred interpreting user preferences: {e}"


augmented_results = augment_listings(results, ideal_listing)
print("\nAugmented Results:")
for r in augmented_results:
    print("Listing:")
    print("Neighborhood:", r["Neighborhood"])
    print("Price:", r["Price"])
    print("Bedrooms:", r["Bedrooms"])
    print("Bathrooms:", r["Bathrooms"])
    print("House Size:", r["House Size"])
    print("Description:", r["Description"])
    print("Neighborhood Description:", r["Neighborhood Description"])
    print("\n")
