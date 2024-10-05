import os
import re
import hashlib
import logging 
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from pinecone import Pinecone
import requests
import aiohttp
import base64
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "ht"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_ENVIRONMENT"] = PINECONE_ENVIRONMENT
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_TOKEN

# ... (rest of the code remains the same)

@app.post("/text-to-speech")
async def text_to_speech(text: str):
    url = "https://api.sarvam.ai/text-to-speech"

    payload = {
        "inputs": [text],
        "target_language_code": "hi-IN",
        "speaker": "meera",
        "pitch": 0,
        "pace": 1.65,
        "loudness": 1.5,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        logger.debug(f"Response status code: {response.status_code}")
        logger.debug(f"Response headers: {response.headers}")
        logger.debug(f"Response content (first 100 chars): {response.text[:100]}")

        return {"audio": response.text, "status": "success"}

    except requests.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return {"error": f"Request failed: {str(e)}", "status": "error"}

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}", "status": "error"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)