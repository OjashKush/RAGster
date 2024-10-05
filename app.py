import streamlit as st
import requests
import json
import base64
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio
import nest_asyncio
from fastapi.middleware.cors import CORSMiddleware

nest_asyncio.apply()

st.set_page_config(page_title="RAG Comparison App", page_icon="🤖")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/query")
async def rag_query(query: Query):
    standard_response = {
        "answer": f"Standard RAG answer to: {query.question}",
        "sources": ["Source 1", "Source 2"],
        "time": 0.5
    }
    agentic_response = {
        "answer": f"Agentic RAG answer to: {query.question}",
        "sources": ["Source A", "Source B"],
        "time": 0.7
    }
    speech_data = {
        "audios": ["base64_encoded_audio_data_here"]
    }
    return {
        "standard_rag": {
            "response": standard_response,
            "time": standard_response["time"],
            "speech": json.dumps(speech_data)
        },
        "agentic_rag": {
            "response": agentic_response,
            "time": agentic_response["time"]
        }
    }

def streamlit_app():
    st.title("RAG Comparison: Standard vs Agentic")
    question = st.text_input("Enter your question:", "Who is Heinrich Rudolph Hertz?")
    if st.button("Submit"):
        with st.spinner("Processing..."):
            result = asyncio.run(rag_query(Query(question=question)))
        
        for rag_type in ["standard_rag", "agentic_rag"]:
            st.subheader(f"{rag_type.replace('_', ' ').title()}")
            st.write(f"Answer: {result[rag_type]['response']['answer']}")
            st.write(f"Time taken: {result[rag_type]['time']:.6f} seconds")
            st.write("Sources:")
            for source in result[rag_type]['response']['sources']:
                st.write(f"- {source}")
        
        st.subheader("Text-to-Speech")
        if result['standard_rag'].get('speech'):
            audio_data = result['standard_rag']['speech']
            audio_bytes = base64.b64decode(json.loads(audio_data)['audios'][0])
            audio_io = io.BytesIO(audio_bytes)
            st.audio(audio_io, format='audio/wav')
            st.success("Audio generated successfully!")
        else:
            st.error("No audio data found for the Standard RAG answer.")
        
        st.write("Debug Information:")
        st.json(result)

async def run():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    api = asyncio.create_task(server.serve())
    streamlit = asyncio.create_task(streamlit_app())
    await asyncio.gather(api, streamlit)

if __name__ == "__main__":
    asyncio.run(run())