import os
from typing import List, Dict
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
        self.document_metadata = {}

    def clear_document_state(self):
        """Clear all document-related state."""
        self.document_chunks = []
        self.chunk_embeddings = None
        self.bm25 = None
        self.index = None
        self.active_files = set()

    def get_document_summary(self, file_path: str) -> Dict:
        """Generate a comprehensive summary of a legal document."""
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                pdf = PdfReader(file)
                text = ''
                for page in pdf.pages:
                    text += page.extract_text()

            prompt = """You are an expert legal assistant with extensive experience in various types of legal documents. Please provide a comprehensive analysis of this legal document following these steps:

            1. Document Identification and Classification:
               - Identify the type of legal document (e.g., contract, court ruling, legislation, brief, etc.)
               - State the jurisdiction and governing law (if applicable)
               - Identify the date and relevant timeline

            2. Key Participants and Roles:
               - List all relevant parties, entities, or institutions involved
               - Describe their respective roles and relationships
               - Identify the authority issuing the document (if applicable)

            3. Core Content Analysis:
               - Main purpose or objective of the document
               - Key findings, decisions, or determinations
               - Critical arguments or reasoning presented
               - Essential terms or conditions
               - Monetary values or financial implications (if any)

            4. Legal Implications:
               - Binding obligations created
               - Rights granted or restricted
               - Precedents cited or established
               - Jurisdictional implications
               - Enforcement mechanisms

            5. Critical Elements:
               - Deadlines, time limits, or important dates
               - Prerequisites or conditions
               - Compliance requirements
               - Procedural requirements
               - Required actions or next steps

            6. Risk Assessment:
               - Potential legal risks or vulnerabilities
               - Compliance challenges
               - Enforcement risks
               - Potential conflicts or ambiguities
               - Missing or inadequate provisions

            7. Recommendations and Notes:
               - Key points requiring attention
               - Suggested actions or responses
               - Areas requiring clarification
               - Additional considerations

            Document text:
            {text}

            Please provide a clear, structured analysis that focuses on the most relevant aspects for this specific type of legal document. Highlight any unusual or particularly significant elements."""

            response = self.llm.invoke(prompt.format(text=text))
            return {
                'summary': response.content,
                'file_path': file_path
            }

    def process_documents(self, file_paths: List[str]):
        """Optimized document processing with reduced token usage."""
        try:
            print(f"Processing documents: {file_paths}")  # Debug log
            self.clear_document_state()
            self.active_files = set(file_paths)

            all_text = []
            
            for file_path in file_paths:
                print(f"Processing file: {file_path}")  # Debug log
                if file_path.endswith('.pdf'):
                    try:
                        with open(file_path, 'rb') as file:
                            pdf = PdfReader(file)
                            text = ''
                            for page in pdf.pages:
                                text += page.extract_text()
                            if text.strip():  # Check if we got any text
                                all_text.append(text)
                                print(f"Successfully extracted text from {file_path} (length: {len(text)})")
                            else:
                                print(f"No text extracted from {file_path}")
                    except Exception as e:
                        print(f"Error reading PDF {file_path}: {str(e)}")
                        continue

            if not all_text:
                print("No text was extracted from documents")
                return False

            print(f"Total text length: {sum(len(t) for t in all_text)}")  # Debug log

            # Split documents into chunks
            self.document_chunks = []
            for text in all_text:
                chunks = self.text_splitter.split_text(text)
                self.document_chunks.extend(chunks)

            print(f"Created {len(self.document_chunks)} chunks")  # Debug log

            if not self.document_chunks:
                print("No chunks created")
                return False

            # Create contextual chunks
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

            print(f"Created {len(contextual_chunks)} contextual chunks")  # Debug log

            # Generate embeddings
            print("Generating embeddings...")  # Debug log
            embeddings = self.embeddings.embed_documents(contextual_chunks)
            self.chunk_embeddings = np.array(embeddings)
            print(f"Generated embeddings with shape: {self.chunk_embeddings.shape}")  # Debug log

            # Create FAISS index
            dimension = self.chunk_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.chunk_embeddings)

            # Create BM25 index
            tokenized_chunks = [chunk.split() for chunk in contextual_chunks]
            self.bm25 = BM25Okapi(tokenized_chunks)

            print("Document processing completed successfully")  # Debug log
            return True

        except Exception as e:
            print(f"Error in process_documents: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full stack trace
            return False

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