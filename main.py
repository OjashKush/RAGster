import os
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from pinecone import Pinecone
import requests
import time
import json
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ht")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# Validate required environment variables
required_env_vars = ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "HUGGINGFACE_API_TOKEN", "SARVAM_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

class Query(BaseModel):
    question: str

class StandardRAG:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
        self.greeting_patterns = [
            r"^(hello|hi|hey|good\s+(morning|afternoon|evening))",
            r"^how are you",
            r"^what's up",
            r"^greetings",
            r"^hola",
            r"^namaste"
        ]
        logger.debug("Standard RAG initialized")

    def is_greeting(self, query: str) -> bool:
        query = query.lower().strip()
        for pattern in self.greeting_patterns:
            if re.match(pattern, query, re.IGNORECASE):
                return True
        return False

    def generate_greeting_response(self, query: str) -> Dict[str, Any]:
        query = query.lower().strip()
        if re.match(r"^how are you", query):
            return {
                "answer": "I'm doing well, thank you for asking! How can I assist you today?",
                "sources": []
            }
        elif re.match(r"^what's up", query):
            return {
                "answer": "Not much, just here to help! What can I do for you?",
                "sources": []
            }
        elif re.match(r"^good\s+(morning|afternoon|evening)", query):
            time_of_day = re.search(r"(morning|afternoon|evening)", query).group(1)
            return {
                "answer": f"Good {time_of_day}! How may I assist you today?",
                "sources": []
            }
        else:
            return {
                "answer": "Hello! How can I assist you today?",
                "sources": []
            }

    def process_query(self, query: str) -> Dict[str, Any]:
        logger.debug(f"Processing query: {query}")
        
        if self.is_greeting(query):
            logger.debug("Greeting detected, generating greeting response")
            return self.generate_greeting_response(query)
        
        try:
            logger.debug("Processing non-greeting query through RAG chain")
            response = self.rag_chain({"query": query})
            answer = response['result'].split("Helpful Answer:")[-1].strip()
            return {
                "answer": answer,
                "sources": [doc.page_content[:100] + "..." for doc in response['source_documents']]
            }
        except Exception as e:
            logger.error(f"Error in StandardRAG query: {str(e)}")
            return {
                "answer": f"An error occurred: {str(e)}. Please try again or contact support.",
                "sources": []
            }

class ReflectiveAgent:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
        self.greeting_patterns = [
            r"^(hello|hi|hey|good\s+(morning|afternoon|evening))",
            r"^how are you",
            r"^what's up",
            r"^greetings",
            r"^hola",
            r"^namaste"
        ]
        logger.debug("Reflective Agent initialized")

    def is_greeting(self, query: str) -> bool:
        query = query.lower().strip()
        logger.debug(f"Checking if '{query}' is a greeting")
        
        for pattern in self.greeting_patterns:
            if re.match(pattern, query, re.IGNORECASE):
                logger.debug(f"Matched greeting pattern: {pattern}")
                return True
        
        logger.debug("Not a greeting")
        return False

    def generate_greeting_response(self, query: str) -> Dict[str, Any]:
        query = query.lower().strip()
        logger.debug(f"Generating greeting response for: {query}")
        
        if re.match(r"^how are you", query):
            response = {
                "answer": "I'm doing well, thank you for asking! I'm here to help with more thoughtful and reflective answers. What can I assist you with?",
                "sources": [],
                "reflection": "Greeting acknowledged with a focus on highlighting reflective capabilities"
            }
        elif re.match(r"^what's up", query):
            response = {
                "answer": "I'm here and ready to help with careful, reflective responses! What would you like to explore?",
                "sources": [],
                "reflection": "Casual greeting acknowledged while maintaining professional tone"
            }
        elif re.match(r"^good\s+(morning|afternoon|evening)", query):
            time_of_day = re.search(r"(morning|afternoon|evening)", query).group(1)
            response = {
                "answer": f"Good {time_of_day}! I'm ready to provide thoughtful assistance with your questions.",
                "sources": [],
                "reflection": "Time-appropriate greeting with emphasis on analytical approach"
            }
        else:
            response = {
                "answer": "Hello! I'm ready to help with careful, reflective answers to your questions.",
                "sources": [],
                "reflection": "Standard greeting with focus on analytical capabilities"
            }
        
        logger.debug(f"Generated greeting response: {response}")
        return response

    def process_query(self, query: str) -> Dict[str, Any]:
        logger.debug(f"Agent processing query: {query}")
        
        if self.is_greeting(query):
            logger.debug("Greeting detected, generating reflective greeting response")
            return self.generate_greeting_response(query)
        
        try:
            logger.debug("Processing non-greeting query with reflection")
            initial_response = self.rag_chain({"query": query})
            initial_answer = initial_response['result'].split("Helpful Answer:")[-1].strip()
            
            reflection_prompt = f"Reflect on and improve this answer: '{initial_answer}'"
            improved_response = self.rag_chain({"query": reflection_prompt})
            improved_answer = improved_response['result'].split("Helpful Answer:")[-1].strip()
            
            return {
                "answer": improved_answer,
                "sources": [doc.page_content[:100] + "..." for doc in initial_response['source_documents']],
                "reflection": "This answer has been refined through self-reflection."
            }
        except Exception as e:
            logger.error(f"Error in ReflectiveAgent query: {str(e)}")
            return {
                "answer": f"An error occurred: {str(e)}. Please try again or contact support.",
                "sources": [],
                "reflection": "An error occurred during the reflection process."
            }

def setup_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = LangchainPinecone.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        text_key="text"
    )
    llm = HuggingFaceHub(
        repo_id="microsoft/Phi-3-mini-4k-instruct",
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

# Global variables
rag_chain = None
standard_rag = None
reflective_agent = None

@app.on_event("startup")
async def startup_event():
    global rag_chain, standard_rag, reflective_agent
    try:
        # Setup RAG chain
        rag_chain = setup_rag_chain()
        logger.info("RAG chain created")

        # Create Standard RAG
        standard_rag = StandardRAG(rag_chain)
        logger.info("Standard RAG created")

        # Create Agentic RAG (Reflective Agent)
        reflective_agent = ReflectiveAgent(rag_chain)
        logger.info("Reflective Agent created")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.post("/query")
async def rag_query(query: Query):
    logger.debug(f"Received query: {query.question}")
    if standard_rag is None or reflective_agent is None:
        raise HTTPException(status_code=500, detail="RAG systems not initialized")
    
    comparison_results = compare_rag_methods(query.question, standard_rag, reflective_agent)
    
    # Generate speech only for the standard_rag response
    try:
        standard_rag_answer = comparison_results["standard_rag"]["response"]["answer"]
        speech_response = await text_to_speech(standard_rag_answer)
        
        if speech_response["status"] == "success":
            comparison_results["standard_rag"]["speech"] = speech_response["audio"]
        else:
            comparison_results["standard_rag"]["speech"] = f"Error generating speech: {speech_response['error']}"
    except Exception as e:
        logger.error(f"Error processing speech for standard_rag: {str(e)}")
        comparison_results["standard_rag"]["speech"] = f"Error processing speech: {str(e)}"
    
    # We're not generating speech for agentic_rag, so we'll just set it to None or remove it
    comparison_results["agentic_rag"].pop("speech", None)
    
    return comparison_results

def compare_rag_methods(query: str, standard_rag: StandardRAG, agentic_rag: ReflectiveAgent) -> Dict[str, Dict[str, Any]]:
    logger.debug(f"Comparing RAG methods for query: {query}")
    
    # Measure time for Standard RAG
    start_time = time.time()
    standard_response = standard_rag.process_query(query)
    standard_time = time.time() - start_time

    # Measure time for Agentic RAG
    start_time = time.time()
    agentic_response = agentic_rag.process_query(query)
    agentic_time = time.time() - start_time

    return {
        "standard_rag": {
            "response": standard_response,
            "time": standard_time
        },
        "agentic_rag": {
            "response": agentic_response,
            "time": agentic_time
        }
    }

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