# README.md

## Workflow Overview
This application processes PDF documents, extracts and indexes text for efficient question answering, and offers a comparison between standard retrieval-based QA (RAG) and a reflective agent (RAG with self-reflection). Additionally, it supports generating speech output for the answers using a text-to-speech (TTS) service and provides a user-friendly interface via Streamlit.

## Key Features

### PDF to Text Conversion
- The app reads PDF files, extracts the text content, and splits it into manageable chunks to facilitate processing.

### Text Embedding
- The extracted text chunks are embedded using Hugging Face's Sentence Transformers. These embeddings serve as the foundation for efficient text retrieval.

### Indexing with Pinecone
- The embedded vectors are indexed in Pinecone, allowing for quick and scalable similarity searches when answering user queries.

### Question Answering (QA)
- Users can submit questions to the `/query` endpoint.
- Two methods are used for responding to queries:
  - **Standard RAG**: Provides an initial response based on indexed text data.
  - **Reflective Agent (RAG with Self-Reflection)**: Enhances the response by reflecting on it, improving the quality of the answer.

### Response Comparison
- The app compares responses from both agents, logs the results, and provides a detailed comparison. Optionally, the Standard RAG response can be converted into speech.

### Speech Generation
- The Standard RAG response can be transformed into speech using an external text-to-speech API, returning the answer as an audio file.

### Frontend Interface
- The application features a Streamlit frontend, offering a user-friendly interface for interacting with the QA system.

## Embeddings and Pinecone Indexing

### Embeddings
- The app utilizes the SentenceTransformer model from Hugging Face to create embeddings for the extracted text chunks. These embeddings are vital for fast and accurate document retrieval.

### Pinecone Indexing
- The embedded vectors are stored in a Pinecone index, which supports scalable and efficient similarity searches. Each text chunk is hashed to generate a unique ID before being indexed.

## RetrievalQA Chain Setup

### Initialization
- The `setup_rag_chain()` function sets up a RetrievalQA chain using Langchain, leveraging a Pinecone vector store and a Hugging Face model for question answering.

### Model
- The Hugging Face model `microsoft/Phi-3-mini-4k-instruct` is employed to answer questions using the indexed data from Pinecone.

## Agents

### Standard RAG
- This agent manages basic RAG queries and can identify greeting phrases such as "hello" or "how are you."
- It decides whether to query the VectorDB based on the input. For instance, a greeting like "hello" does not prompt a VectorDB query.
- For standard queries, it returns answers based on the indexed data.

### Reflective Agent
- This agent performs an additional step of self-reflection. After generating an initial response, it reflects on the answer to refine and improve it.
- Reflection is inspired by research papers:
  - **Self-Refine**: Iterative Refinement with Self-Feedback (Madaan et al., 2023) [1] https://www.researchgate.net/publication/369740347_Self-Refine_Iterative_Refinement_with_Self-Feedback
  - **Reflexion**: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023). [2] https://arxiv.org/abs/2303.11366

## Images of Streamlit App

Main Interface[1] ![main](https://github.com/OjashKush/RAGster/blob/main/ragstermain.jpeg)

Audio Generation[2] ![audio](https://github.com/OjashKush/RAGster/blob/main/ragsteraudio.jpeg)

Images of the app interface are available for demonstration purposes.

## Getting Started

### Prerequisites
- Python 3.x

### Install Dependencies
Install the required dependencies using:
```bash
pip install -r requirements.txt
```

### Running the App

1. **Start the FastAPI server:**
   ```bash
   uvicorn app:app --reload
   ```

2. **Launch the Streamlit Frontend:**
   ```bash
   streamlit run streamlit_app.py
   ```

Enjoy exploring your PDF documents with advanced question answering capabilities!
