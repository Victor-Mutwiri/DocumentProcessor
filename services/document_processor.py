import os
import re
from typing import List, Dict, Any
import groq
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np
import faiss
from pypdf import PdfReader
from docx import Document
import win32com.client as win32

class DocumentProcessor:
    def __init__(self):
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama-3.3-70b-versatile"
        )
        self.text_splitter = CharacterTextSplitter(
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
        self.document_metadata = {}
        self.processing_status = "Idle"

    def clear_document_state(self):
        """Clear all document-related state."""
        self.document_chunks = []
        self.chunk_embeddings = None
        self.bm25 = None
        self.index = None
        self.active_files = set()
        self.processing_status = "Idle"

    def get_document_summary(self, file_path: str) -> Dict:
        """Generate an intelligent legal analysis of the document."""
        try:
            if not file_path.endswith('.pdf'):
                raise ValueError("Only PDF files are supported")

            print(f"Generating analysis for: {file_path}")
            
            with open(file_path, 'rb') as file:
                pdf = PdfReader(file)
                text = ''
                for page in pdf.pages:
                    text += page.extract_text()
                
                if not text.strip():
                    raise ValueError("No text could be extracted from the PDF")

                # Use smaller chunks for better token management
                max_chunk_size = 2500  # Reduced size for better token control
                chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
                print(f"Analyzing document in {len(chunks)} segments")

                summaries = []
                for i, chunk in enumerate(chunks):
                    chunk_prompt = f"""As an expert legal assistant, examine this document segment and provide crucial insights. 
                    This is segment {i+1} of {len(chunks)}.

                    Document Text:
                    {chunk}

                    Analyze this segment with strategic focus on:
                    - Key legal principles and their implications
                    - Critical factual or procedural elements
                    - Strategic considerations and potential impacts
                    
                    Focus on what's most significant in this segment, adapting your analysis to the document's nature and context.
                    Be concise but thorough, highlighting only what's truly important."""

                    print(f"Processing segment {i+1}/{len(chunks)}")
                    response = self.llm.invoke(chunk_prompt)
                    summaries.append(response.content)

                # Synthesize the analysis with a strategic focus
                synthesis_prompt = """Synthesize these analytical segments into a strategic legal intelligence brief:

                Previous Analyses:
                {}

                Provide an executive-level analysis that:
                1. Identifies the document's strategic significance and core implications
                2. Highlights critical elements requiring attention or action
                3. Outlines potential opportunities, risks, and strategic considerations
                4. Provides context-aware, actionable insights

                Focus on delivering practical, strategic value while maintaining analytical rigor.""".format(
                    "\n---\n".join(summaries[:3])  # Limit to prevent token overflow
                )

                final_response = self.llm.invoke(synthesis_prompt)
                
                print("Analysis complete")
                return {
                    'summary': final_response.content,
                    'file_path': file_path
                }

        except Exception as e:
            print(f"Error in document analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def process_documents(self, file_paths: List[str]):
        try:
            self.processing_status = "Processing selected documents"
            print(f"Processing documents: {file_paths}")
            self.clear_document_state()
            self.active_files = set(file_paths)

            all_text = []
            
            for file_path in file_paths:
                print(f"Processing file: {file_path}")
                if file_path.endswith('.pdf'):
                    try:
                        with open(file_path, 'rb') as file:
                            pdf = PdfReader(file)
                            text = ''
                            for page in pdf.pages:
                                text += page.extract_text()
                            if text.strip():
                                all_text.append(text)
                                print(f"Successfully extracted text from {file_path} (length: {len(text)})")
                            else:
                                print(f"No text extracted from {file_path}")
                    except Exception as e:
                        print(f"Error reading PDF {file_path}: {str(e)}")
                        continue
                elif file_path.endswith('.docx'):
                    try:
                        doc = Document(file_path)
                        text = '\n'.join([para.text for para in doc.paragraphs])
                        if text.strip():
                            all_text.append(text)
                            print(f"Successfully extracted text from {file_path} (length: {len(text)})")
                        else:
                            print(f"No text extracted from {file_path}")
                    except Exception as e:
                        print(f"Error reading DOCX {file_path}: {str(e)}")
                        continue
                elif file_path.endswith('.doc'):
                    try:
                        word = win32.Dispatch("Word.Application")
                        doc = word.Documents.Open(file_path)
                        text = doc.Content.Text
                        doc.Close()
                        word.Quit()
                        if text.strip():
                            all_text.append(text)
                            print(f"Successfully extracted text from {file_path} (length: {len(text)})")
                        else:
                            print(f"No text extracted from {file_path}")
                    except Exception as e:
                        print(f"Error reading DOC {file_path}: {str(e)}")
                        continue

            if not all_text:
                print("No text was extracted from documents")
                self.processing_status = "No text extracted from documents"
                return False

            print(f"Total text length: {sum(len(t) for t in all_text)}")

            self.document_chunks = []
            for text in all_text:
                chunks = self.text_splitter.split_text(text)
                self.document_chunks.extend(chunks)

            print(f"Created {len(self.document_chunks)} chunks")

            if not self.document_chunks:
                print("No chunks created")
                self.processing_status = "No chunks created"
                return False

            contextual_chunks = []
            for i, chunk in enumerate(self.document_chunks):
                context = f"Document chunk {i+1}: This section appears to contain information about "
                if "WHEREAS" in chunk or "NOW, THEREFORE" in chunk:
                    context += "recitals or preliminary statements. "
                elif "SHALL" in chunk or "MUST" in chunk:
                    context += "obligations or requirements. "
                elif "GOVERNING LAW" in chunk:
                    context += "jurisdiction and applicable law. "
                
                contextual_chunk = context + chunk
                contextual_chunks.append(contextual_chunk)

            print(f"Created {len(contextual_chunks)} contextual chunks")

            print("Generating embeddings...")
            embeddings = self.embeddings.embed_documents(contextual_chunks)
            self.chunk_embeddings = np.array(embeddings)
            print(f"Generated embeddings with shape: {self.chunk_embeddings.shape}")

            dimension = self.chunk_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.chunk_embeddings)

            tokenized_chunks = [chunk.split() for chunk in contextual_chunks]
            self.bm25 = BM25Okapi(tokenized_chunks)

            print("Document processing completed successfully")
            self.processing_status = "Document processing completed successfully"
            return True

        except Exception as e:
            print(f"Error in process_documents: {str(e)}")
            import traceback
            traceback.print_exc()
            self.processing_status = f"Error in process_documents: {str(e)}"
            return False

    def get_processing_status(self):
        return self.processing_status

    def chat_with_documents(self, query: str) -> str:
        try:
            """Optimized chat functionality with comprehensive analysis."""
            if not self.document_chunks:
                return "No documents have been processed yet."

            query_embedding = self.embeddings.embed_query(query)
            k = 3
            D, I = self.index.search(np.array([query_embedding]), k)
            bm25_scores = self.bm25.get_scores(query.split())
            top_bm25_indices = np.argsort(bm25_scores)[-k:][::-1]

            relevant_chunks = set()
            for idx in I[0]:
                relevant_chunks.add(self.document_chunks[idx])
            for idx in top_bm25_indices:
                relevant_chunks.add(self.document_chunks[idx])

            context = "\n\n".join(list(relevant_chunks)[:k])
            prompt = f"""As a legal expert, provide a thorough and nuanced analysis of the following question based on the given context. 
            Consider all relevant legal aspects, implications, and interconnected issues. 
            Draw from the context to support your analysis, but also highlight any important considerations or limitations.
            Ensure your response is clear, precise, and comprehensive while remaining directly relevant to the query.

            Context:
            {context}

            Question: {query}

            Provide a well-reasoned response that addresses all pertinent aspects of the question. If there are multiple interpretations or considerations possible, include them. If certain information is missing or unclear, acknowledge this and explain its potential impact on the analysis."""

            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            print(f"Error in document chat: {e}")
            return "An error occurred while processing your request."