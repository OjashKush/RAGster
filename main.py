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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PINECONE_API_KEY = "8771b271-b1d0-4e66-98d1-1dda1d701f62"
PINECONE_ENVIRONMENT = "your-environment"
INDEX_NAME = "ht"
HUGGINGFACE_API_TOKEN = "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"

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_ENVIRONMENT"] = PINECONE_ENVIRONMENT
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_TOKEN

def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, 'rb') as file:
        pdf = PdfReader(file)
        return ''.join(page.extract_text() for page in pdf.pages)

def process_pdfs(file_or_directory: str) -> List[str]:
    texts = []
    if os.path.isfile(file_or_directory) and file_or_directory.endswith('.pdf'):
        texts.append(extract_text_from_pdf(file_or_directory))
    elif os.path.isdir(file_or_directory):
        for filename in os.listdir(file_or_directory):
            if filename.endswith('.pdf'):
                file_path = os.path.join(file_or_directory, filename)
                texts.append(extract_text_from_pdf(file_path))
    else:
        raise FileNotFoundError(f"No such file or directory: '{file_or_directory}'")
    return texts

def split_texts(texts: List[str]) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    return [chunk for text in texts for chunk in text_splitter.split_text(text)]

class SentenceTransformerEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text)

def index_documents(chunks: List[str], embeddings: HuggingFaceEmbeddings):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    for chunk in chunks:
        embedding = embeddings.embed_query(chunk)
        metadata = {"text": chunk}
        id = hashlib.md5(chunk.encode()).hexdigest()
        index.upsert(vectors=[{"id": id, "values": embedding, "metadata": metadata}])

def setup_rag_chain():
    # Change the embedding model to match the index dimension (384)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = LangchainPinecone.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        text_key="text"
    )
    llm = HuggingFaceHub(repo_id="microsoft/Phi-3-mini-4k-instruct")
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

def query_rag(rag_chain, question: str) -> Dict[str, str]:
    logger.debug(f"Querying RAG with: {question}")
    try:
        response = rag_chain({"query": question})
        logger.debug(f"RAG response: {response}")
        
        if 'result' not in response or 'source_documents' not in response:
            logger.error(f"Unexpected RAG response structure: {response}")
            raise ValueError("Unexpected RAG response structure")

        if response['result'].strip().startswith("Use the following pieces of context"):
            logger.error(f"RAG returned default prompt: {response['result']}")
            raise ValueError("RAG returned default prompt")

        return {
            "answer": response['result'],
            "sources": [doc.page_content for doc in response['source_documents']]
        }
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}")
        return {
            "answer": f"An error occurred: {str(e)}. Please try again or contact support.",
            "sources": []
        }
    

class StandardRAG:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
        self.greeting_patterns = [
            r"^(hello|hi|hey|good\s+(morning|afternoon|evening))",
            r"^how are you",
            r"^what's up",
        ]
        logger.debug("Standard RAG initialized")

    def is_greeting(self, query: str) -> bool:
        query = query.lower().strip()
        logger.debug(f"Checking if '{query}' is a greeting")
    
        for pattern in self.greeting_patterns:
            if re.match(pattern, query):
                logger.debug(f"Matched greeting pattern: {pattern}")
                return True
    
        logger.debug("Not a greeting")
        return False

    def generate_greeting_response(self, query: str) -> str:
        logger.debug(f"Generating greeting response for: {query}")
        if re.match(r"^how are you", query.lower()):
            return "I'm doing well, thank you for asking! How can I assist you today?"
        elif re.match(r"^what's up", query.lower()):
            return "Not much, just here to help! What can I do for you?"
        else:
            return "Hello! How can I assist you today?"

    def process_query(self, query: str) -> Dict[str, any]:
        logger.debug(f"StandardRAG processing query: {query}")
        
        # Check if the query is a greeting
        if self.is_greeting(query):
            response = self.generate_greeting_response(query)
            logger.debug(f"Greeting response generated: {response}")
            return {"answer": response, "sources": []}
        
        # Proceed with querying RAG for non-greetings
        try:
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
        ]
        logger.debug("Agent initialized")

    def is_greeting(self, query: str) -> bool:
        query = query.lower().strip()
        logger.debug(f"Checking if '{query}' is a greeting")
    
        for pattern in self.greeting_patterns:
            if re.match(pattern, query):
                logger.debug(f"Matched greeting pattern: {pattern}")
                return True
    
        logger.debug("Not a greeting")
        return False

    def generate_greeting_response(self, query: str) -> str:
        logger.debug(f"Generating greeting response for: {query}")
        if re.match(r"^how are you", query.lower()):
            return "I'm doing well, thank you for asking! How can I assist you today?"
        elif re.match(r"^what's up", query.lower()):
            return "Not much, just here to help! What can I do for you?"
        else:
            return "Hello! How can I assist you today?"

    def process_query(self, query: str) -> Dict[str, any]:
        logger.debug(f"Agent processing query: {query}")

        if self.is_greeting(query):
            response = self.generate_greeting_response(query)
            logger.debug(f"Greeting response generated: {response}")
            return {"answer": response, "sources": []}  # No reflection needed for greetings
        else:
            logger.debug("Not a greeting, querying RAG")
            return self.query_rag_with_reflection(query)  # Call reflection query

    def query_rag_with_reflection(self, query: str) -> Dict[str, any]:
        try:
            initial_response = self.rag_chain({"query": query})
            initial_answer = initial_response['result'].split("Helpful Answer:")[-1].strip()

            # Only reflect if the initial answer isn't a greeting
            if not self.is_greeting(initial_answer):  # Check if the response is a greeting
                reflection_prompt = f"Reflect on and improve this answer: '{initial_answer}'"
                improved_response = self.rag_chain({"query": reflection_prompt})
                improved_answer = improved_response['result'].split("Helpful Answer:")[-1].strip()

                return {
                    "answer": improved_answer,
                    "sources": [doc.page_content[:100] + "..." for doc in initial_response['source_documents']],
                    "reflection": "This answer has been refined through self-reflection."
                }
            else:
                # If the initial answer is a greeting, just return it
                return {
                    "answer": initial_answer,
                    "sources": [doc.page_content[:100] + "..." for doc in initial_response['source_documents']],
                    "reflection": None
                }

        except Exception as e:
            logger.error(f"Error in ReflectiveAgent query: {str(e)}")
            return {
                "answer": f"An error occurred: {str(e)}. Please try again or contact support.",
                "sources": [],
                "reflection": "An error occurred during the reflection process."
            }

def compare_rag_methods(query: str, standard_rag: StandardRAG, agentic_rag: ReflectiveAgent) -> Dict[str, Dict[str, str]]:
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

app = FastAPI()

class Query(BaseModel):
    question: str

# Global variables
rag_chain = None
standard_rag = None
reflective_agent = None
MAX_TEXT_LENGTH = 500

@app.on_event("startup")
async def startup_event():
    global rag_chain, standard_rag, reflective_agent
    try:
        # ... (keep the existing startup logic)

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


import aiohttp
import base64
import json
from fastapi import HTTPException

@app.post("/text-to-speech")
async def text_to_speech(text: str):
    url = "https://api.sarvam.ai/text-to-speech"

    payload = {
        "inputs": [text],  # This is where the standard_rag answer goes
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
        "api-subscription-key": "0b557932-7b2d-424b-a365-dc7e3ecc97ba",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        logger.debug(f"Response status code: {response.status_code}")
        logger.debug(f"Response headers: {response.headers}")
        logger.debug(f"Response content (first 100 chars): {response.text[:100]}")

        # The response is expected to be the audio data directly
        return {"audio": response.text, "status": "success"}

    except requests.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return {"error": f"Request failed: {str(e)}", "status": "error"}

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}", "status": "error"}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    