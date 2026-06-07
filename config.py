# Config.py file is responsible for loading environment variables, setting up the Neo4j driver, and initializing the Google Generative AI Embeddings. It ensures that all necessary configurations are in place for the application to function correctly.

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings 

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Exception handling for missing API key
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in environment variables. Add it to you .env file")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "AlphaFund2026!")

NEO4J_DRIVER = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
EMBEDDER = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")