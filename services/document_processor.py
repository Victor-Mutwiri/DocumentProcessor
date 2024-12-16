import os
from typing import List
import groq
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np
import faiss
from pypdf import PdfReader

class DocumentProcessor:
    def __init__(self):
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama-3.3-70b-versatile"
            """ model_name="mixtral-8x7b-32768" """
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.active_files = set()
        self.document_chunks = []
        self.chunk_embeddings = None
        self.bm25 = None
        self.index = None

    def clear_document_state(self):
        """Clear all document-related state."""
        self.document_chunks = []
        self.chunk_embeddings = None
        self.bm25 = None
        self.index = None
        self.active_files = set()

    def process_documents(self, file_paths: List[str]):
        """Process uploaded documents and create embeddings and BM25 index."""
        self.clear_document_state()
        self.active_files = set(file_paths)

        all_text = []
        
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                with open(file_path, 'rb') as file:
                    pdf = PdfReader(file)
                    text = ''
                    for page in pdf.pages:
                        text += page.extract_text()
                    all_text.append(text)

        # Split documents into chunks
        self.document_chunks = []
        for text in all_text:
            chunks = self.text_splitter.split_text(text)
            self.document_chunks.extend(chunks)

        # Create contextual chunks
        contextual_chunks = []
        for i, chunk in enumerate(self.document_chunks):
            context = f"Document chunk {i+1}: This excerpt contains information about "
            contextual_chunk = context + chunk
            contextual_chunks.append(contextual_chunk)

        # Generate embeddings
        embeddings = self.embeddings.embed_documents(contextual_chunks)
        self.chunk_embeddings = np.array(embeddings)

        # Create FAISS index
        dimension = self.chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.chunk_embeddings)

        # Create BM25 index
        tokenized_chunks = [chunk.split() for chunk in contextual_chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def chat_with_documents(self, query: str) -> str:
        """Chat with the documents using hybrid search (BM25 + embeddings)."""
        if not self.document_chunks:
            return "No documents have been processed yet."

        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Perform FAISS search
        k = 5  # number of neighbors to retrieve
        D, I = self.index.search(np.array([query_embedding]), k)
        
        # Perform BM25 search
        bm25_scores = self.bm25.get_scores(query.split())
        top_bm25_indices = np.argsort(bm25_scores)[-k:][::-1]

        # Combine results
        relevant_chunks = set()
        for idx in I[0]:
            relevant_chunks.add(self.document_chunks[idx])
        for idx in top_bm25_indices:
            relevant_chunks.add(self.document_chunks[idx])

        # Create context for the LLM
        context = "\n\n".join(list(relevant_chunks))
        prompt = f"""Based on the following context, please answer the question. 
        If the answer cannot be found in the context, say so.

        Context:
        {context}

        Question: {query}

        Answer:"""

        # Get response from LLM
        response = self.llm.invoke(prompt)
        return response.content 