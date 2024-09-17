import streamlit as st
import requests
import json
import base64
import io

# Set page config at the very beginning
st.set_page_config(page_title="RAG Comparison App", page_icon="🤖")

def query_rag_api(question):
    url = "http://localhost:8000/query"  # Replace with your actual API endpoint
    payload = {"question": question}
    response = requests.post(url, json=payload)
    return response.json()

# Streamlit UI 
st.title("RAG Comparison: Standard vs Agentic")

question = st.text_input("Enter your question:", "Who is Heinrich Rudolph Hertz?")

if st.button("Submit"):
    with st.spinner("Processing..."):
        result = query_rag_api(question)

    # Displaying Standard RAG results
    st.subheader("Standard RAG")
    st.write(f"Answer: {result['standard_rag']['response']['answer']}")
    st.write(f"Time taken: {result['standard_rag']['time']:.6f} seconds")
    st.write("Sources:")
    for source in result['standard_rag']['response']['sources']:
        st.write(f"- {source}")

    # Displaying Agentic RAG results
    st.subheader("Agentic RAG")
    st.write(f"Answer: {result['agentic_rag']['response']['answer']}")
    st.write(f"Time taken: {result['agentic_rag']['time']:.6f} seconds")
    st.write("Sources:")
    for source in result['agentic_rag']['response']['sources']:
        st.write(f"- {source}")

    # Handle Text-to-Speech (TTS) for the Standard RAG answer
    st.subheader("Text-to-Speech")
    
    if result['standard_rag'].get('speech'):
        # Decode the base64-encoded audio data
        audio_data = result['standard_rag']['speech']
        audio_bytes = base64.b64decode(json.loads(audio_data)['audios'][0])
        
        # Create a BytesIO object for playing the audio
        audio_io = io.BytesIO(audio_bytes)
        
        # Play the audio using st.audio
        st.audio(audio_io, format='audio/wav')
        st.success("Audio generated successfully!")
    else:
        st.error("No audio data found for the Standard RAG answer.")
    
    st.write("Debug Information:")
    st.json(result)

if __name__ == "__main__":
    pass  # The app is already running at this point
