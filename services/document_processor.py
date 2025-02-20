import os
import re
import torch
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModel
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from transformers import pipeline
import numpy as np
import faiss
from pypdf import PdfReader

class LegalDocumentSplitter:
    """Custom text splitter optimized for legal documents"""
    def __init__(self):
        # Legal document section headers
        self.section_patterns = [
            r"(?i)^Article \d+",
            r"(?i)^Section \d+",
            r"(?i)^Clause \d+",
            r"(?i)^WHEREAS",
            r"(?i)^NOW, THEREFORE",
            r"(?i)^IN WITNESS WHEREOF",
            r"(?i)^APPENDIX [A-Z]",
            r"(?i)^Schedule \d+",
            r"(?i)^EXHIBIT [A-Z]"
        ]
        
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
        
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

    def split_text(self, text: str) -> List[str]:
        # First, identify major sections using legal patterns
        sections = []
        current_section = []
        
        for line in text.split('\n'):
            if any(re.match(pattern, line.strip()) for pattern in self.section_patterns):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))

        # If no sections were identified, try markdown-style splitting
        if len(sections) <= 1:
            try:
                markdown_splits = self.markdown_splitter.split_text(text)
                sections = [doc.page_content for doc in markdown_splits]
            except:
                sections = [text]

        # Further split large sections while preserving legal context
        final_chunks = []
        for section in sections:
            if len(section.split()) > 200:  # If section is too large
                sub_chunks = self.recursive_splitter.split_text(section)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(section)

        return final_chunks

class LegalBERTEmbeddings:
    def __init__(self):
        try:
            self.model_name = "nlpaueb/legal-bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Change NER model to a more compatible one
            self.ner = pipeline("ner", model="dslim/bert-base-NER")
            
            # Add cross-encoder for re-ranking
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
        except Exception as e:
            print(f"Error initializing LegalBERTEmbeddings: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Enhanced preprocessing for legal text"""
        # Normalize section references
        text = re.sub(r'(?i)section(?=\s+\d)', 'Section', text)
        text = re.sub(r'(?i)article(?=\s+\d)', 'Article', text)
        
        # Standardize legal citations
        text = re.sub(r'(\d+)\s*U\.?S\.?C\.?\s*§*\s*(\d+)', r'\1 USC § \2', text)
        
        return text

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        embeddings = []
        for doc in documents:
            # Preprocess the text
            processed_doc = self.preprocess_text(doc)
            
            # Extract legal entities
            entities = self.ner(processed_doc)
            
            # Add entity information to the embedding context
            entity_context = " ".join([ent['word'] for ent in entities])
            enhanced_doc = f"{processed_doc} {entity_context}"
            
            # Generate embeddings
            inputs = self.tokenizer(enhanced_doc, 
                                  return_tensors="pt", 
                                  padding=True, 
                                  truncation=True, 
                                  max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                embeddings.append(cls_embedding)
        
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        processed_query = self.preprocess_text(query)
        inputs = self.tokenizer(processed_query, 
                              return_tensors="pt", 
                              padding=True, 
                              truncation=True, 
                              max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        return cls_embedding

class EnhancedDocumentProcessor:
    def __init__(self):
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.4
        )
        self.text_splitter = LegalDocumentSplitter()
        self.embeddings = LegalBERTEmbeddings()
        self.document_chunks = []
        self.chunk_embeddings = None
        self.index = None
        self.chunk_metadata = []

    def process_documents(self, file_paths: List[str]) -> bool:
        try:
            all_text = []
            
            for file_path in file_paths:
                if file_path.endswith('.pdf'):
                    with open(file_path, 'rb') as file:
                        pdf = PdfReader(file)
                        text = ''
                        for page in pdf.pages:
                            text += page.extract_text()
                        if text.strip():
                            # Store metadata with the text
                            all_text.append({
                                'text': text,
                                'source': file_path,
                                'type': 'pdf'
                            })

            if not all_text:
                return False

            self.document_chunks = []
            self.chunk_metadata = []
            
            for doc in all_text:
                chunks = self.text_splitter.split_text(doc['text'])
                for chunk in chunks:
                    self.document_chunks.append(chunk)
                    self.chunk_metadata.append({
                        'source': doc['source'],
                        'type': doc['type'],
                        'length': len(chunk)
                    })

            print(f"Created {len(self.document_chunks)} chunks")
            
            # Generate embeddings with enhanced context
            embeddings = self.embeddings.embed_documents(self.document_chunks)
            self.chunk_embeddings = np.array(embeddings)

            # Initialize FAISS index
            dimension = self.chunk_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.chunk_embeddings)

            return True

        except Exception as e:
            print(f"Error in process_documents: {str(e)}")
            return False

    def chat_with_documents(self, query: str) -> str:
        try:
            if not self.document_chunks:
                return "No documents have been processed yet."

            # Get query embedding and initial candidates
            query_embedding = self.embeddings.embed_query(query)
            k = 5  # Retrieve more candidates for re-ranking
            D, I = self.index.search(np.array([query_embedding]), k)

            # Prepare candidates for re-ranking
            candidates = [(self.document_chunks[idx], score) for idx, score in zip(I[0], D[0])]
            
            # Re-rank using cross-encoder
            cross_encoder_inputs = [(query, doc[0]) for doc in candidates]
            cross_encoder_scores = self.embeddings.cross_encoder.predict(cross_encoder_inputs)
            
            # Sort by cross-encoder scores and take top 3
            reranked_chunks = [candidates[i][0] for i in np.argsort(cross_encoder_scores)[-3:]]
            
            # Enhance context with metadata
            context = ""
            for chunk in reranked_chunks:
                idx = self.document_chunks.index(chunk)
                metadata = self.chunk_metadata[idx]
                context += f"\nSource: {metadata['source']}\n{chunk}\n"

            prompt = f"""As a legal expert, analyze this question based on the provided legal documents:

            Context:
            {context}

            Question: {query}

            Provide a detailed analysis that:
            1. Cites specific sections/clauses when relevant
            2. Explains the legal reasoning behind the answer
            3. Identifies any potential ambiguities or limitations
            4. References the source documents where appropriate"""

            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            print(f"Error in document chat: {e}")
            return "An error occurred while processing your request."
    
    def review_contract(self, file_paths: List[str]) -> str:
        """
        Processes uploaded contract documents and returns a comprehensive review.
        The review covers key terms, obligations, potential risks, hidden pitfalls,
        ambiguous language, and recommendations for further clarification.
        """
        try:
            # Load the contract document(s)
            all_text = []
            for file_path in file_paths:
                if file_path.endswith('.pdf'):
                    with open(file_path, 'rb') as file:
                        pdf = PdfReader(file)
                        text = ''
                        for page in pdf.pages:
                            text += page.extract_text()
                        if text.strip():
                            all_text.append({
                                'text': text,
                                'source': file_path,
                                'type': 'pdf'
                            })

            if not all_text:
                return "No valid contract document found."

            # Split the contract text into manageable chunks
            contract_chunks = []
            contract_metadata = []
            for doc in all_text:
                chunks = self.text_splitter.split_text(doc['text'])
                for chunk in chunks:
                    contract_chunks.append(chunk)
                    contract_metadata.append({
                        'source': doc['source'],
                        'length': len(chunk)
                    })

            print(f"Contract review: created {len(contract_chunks)} chunks.")

            # Generate embeddings for each chunk and build a FAISS index for retrieval
            embeddings = self.embeddings.embed_documents(contract_chunks)
            chunk_embeddings = np.array(embeddings)
            dimension = chunk_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(chunk_embeddings)

            # Define a review query that encapsulates all the critical contract review points
            review_query = (
                "Review this contract document in detail. Provide a comprehensive analysis covering the following: "
                "1. A summary of key terms and obligations. "
                "2. Identification of any overly binding or unfavorable clauses. "
                "3. Potential hidden pitfalls or liabilities that could bind the client unexpectedly. "
                "4. Areas with ambiguous or unclear language that require further clarification. "
                "5. Your recommendation on whether the client should sign the contract, and what clarifications to seek if needed."
            )

            # Retrieve the most relevant chunks using the FAISS index
            query_embedding = self.embeddings.embed_query(review_query)
            k = 5  # Retrieve the top 5 most relevant chunks
            D, I = index.search(np.array([query_embedding]), k)
            candidates = [(contract_chunks[idx], score) for idx, score in zip(I[0], D[0])]

            # Use the cross-encoder to re-rank the candidates for best context
            cross_encoder_inputs = [(review_query, doc[0]) for doc in candidates]
            cross_encoder_scores = self.embeddings.cross_encoder.predict(cross_encoder_inputs)
            top_indices = np.argsort(cross_encoder_scores)[-3:]  # select top 3 chunks
            reranked_chunks = [candidates[i][0] for i in top_indices]

            # Combine the selected chunks with their metadata into a context string
            context = ""
            for chunk in reranked_chunks:
                idx = contract_chunks.index(chunk)
                metadata = contract_metadata[idx]
                context += f"\nSource: {metadata['source']}\n{chunk}\n"

            # Build the detailed prompt for contract review
            prompt = f"""As a legal expert specializing in contracts, review the following contract document and provide a comprehensive analysis including:
                1. A summary of the key terms and obligations.
                2. Identification of any clauses that may be overly binding or unfavorable.
                3. Potential hidden pitfalls or liabilities that could adversely affect the client.
                4. Areas where the language is ambiguous or requires further clarification.
                5. Your recommendation on whether the client should sign the contract, along with detailed reasons.

                Contract Context:
                {context}
                """

            # Invoke the language model with the prompt
            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            print(f"Error in contract review: {e}")
            return "An error occurred while reviewing the contract."
        
