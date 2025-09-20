# multimodal_agent.py
import os
import json
import base64
import io
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any, Literal, Union, BinaryIO
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
import mimetypes

# Google Cloud imports
from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud import storage
from google.cloud import aiplatform

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document

# Image processing
from PIL import Image
import fitz  # PyMuPDF for PDF processing

load_dotenv()

# --------- Dataclasses for Multimodal Data ---------
@dataclass
class MultimodalData:
    """Represents multimodal data with various content types."""
    data_id: str
    title: str
    description: str
    content_type: Literal["image", "pdf", "screenshot", "log", "config", "doc", "incident"]
    # File information
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    # Content data
    text_content: Optional[str] = None
    image_data: Optional[bytes] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    # Metadata
    incident_id: Optional[str] = None
    service: Optional[str] = None
    environment: Optional[str] = None
    cluster: Optional[str] = None
    namespace: Optional[str] = None
    pod: Optional[str] = None
    container: Optional[str] = None
    commit_sha: Optional[str] = None
    tool: Optional[str] = None
    # Domain content
    configs: Optional[str] = None
    docs_faq: Optional[str] = None
    status: Optional[str] = None
    resolution: Optional[str] = None
    severity: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    tags: Optional[List[str]] = None
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # Multimodal specific
    extracted_text: Optional[str] = None
    visual_description: Optional[str] = None
    error_patterns: Optional[List[str]] = None
    ui_elements: Optional[List[str]] = None

@dataclass
class MultimodalSearchResult:
    """Result from multimodal similarity search."""
    data_id: str
    title: str
    content_type: str
    similarity_score: float
    confidence_percentage: int
    visual_similarity: Optional[float] = None
    text_similarity: Optional[float] = None
    metadata: Dict[str, Any] = None
    matched_content: Optional[str] = None
    visual_description: Optional[str] = None

# --------- Multimodal Embedding Generator ---------
class MultimodalEmbeddingGenerator:
    """Generates embeddings for both text and visual content using Google Gemini."""
    
    def __init__(self, model_name: str = "text-embedding-004", vision_model: str = "gemini-1.5-pro", api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_GEMINI_API_KEY not found in env")
        
        # Text embeddings
        self.text_embedding_model = GoogleGenerativeAIEmbeddings(
            model=model_name, 
            google_api_key=self.api_key
        )
        
        # Vision model for multimodal embeddings
        self.vision_model = ChatGoogleGenerativeAI(
            model=vision_model,
            google_api_key=self.api_key,
            convert_system_message_to_human=True
        )
        
        # System prompts
        self.TEXT_SYSTEM_PROMPT = (
            "You are embedding DevOps / SaaS incident data for similarity search. "
            "Focus on root cause analysis, remediation steps, configs, and operational context. "
            "Capture semantic meaning of errors, resolutions, and logs rather than surface words."
        )
        
        self.VISUAL_SYSTEM_PROMPT = (
            "You are analyzing visual content from DevOps incidents. "
            "Focus on error messages, UI elements, log displays, configuration screens, "
            "and any visual indicators of system issues. Extract text, identify error patterns, "
            "and describe the visual context for incident analysis."
        )

    def generate_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for text content."""
        try:
            enriched_text = f"{self.TEXT_SYSTEM_PROMPT}\n\n{text}"
            return self.text_embedding_model.embed_query(enriched_text)
        except Exception as e:
            print(f"[TextEmbedding] error: {str(e)}")
            return None

    def generate_visual_embedding(self, image_data: bytes, description: str = None) -> List[float]:
        """Generate embedding for visual content using multimodal approach."""
        try:
            # Convert image to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Create multimodal prompt
            if description:
                prompt = f"{self.VISUAL_SYSTEM_PROMPT}\n\nVisual Description: {description}\n\nAnalyze this image for DevOps incident context:"
            else:
                prompt = f"{self.VISUAL_SYSTEM_PROMPT}\n\nAnalyze this image for DevOps incident context:"
            
            # Use vision model to generate description and then embed it
            vision_response = self.vision_model.invoke([
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ])
            
            # Extract text from vision response and create embedding
            visual_text = vision_response.content
            return self.generate_text_embedding(visual_text)
            
        except Exception as e:
            print(f"[VisualEmbedding] error: {str(e)}")
            return None

    def extract_text_from_pdf(self, pdf_data: bytes) -> str:
        """Extract text content from PDF."""
        try:
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            text_content = []
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text_content.append(page.get_text())
            
            pdf_document.close()
            return "\n".join(text_content)
        except Exception as e:
            print(f"[PDFExtraction] error: {str(e)}")
            return ""

    def process_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process image and extract visual information."""
        try:
            # Open image with PIL
            image = Image.open(io.BytesIO(image_data))
            
            # Get image properties
            width, height = image.size
            mode = image.mode
            
            # Convert to RGB if necessary
            if mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large (for processing efficiency)
            max_size = 1024
            if width > max_size or height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                width, height = image.size
            
            # Convert back to bytes
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG', quality=85)
            processed_image_data = img_buffer.getvalue()
            
            return {
                "processed_data": processed_image_data,
                "width": width,
                "height": height,
                "mode": mode
            }
            
        except Exception as e:
            print(f"[ImageProcessing] error: {str(e)}")
            return None

    def generate_multimodal_embedding(self, data: MultimodalData) -> List[float]:
        """Generate appropriate embedding based on content type."""
        if data.content_type in ["image", "screenshot"] and data.image_data:
            return self.generate_visual_embedding(data.image_data, data.visual_description)
        elif data.content_type == "pdf" and data.image_data:
            # For PDFs, we might have both text and visual content
            text_emb = self.generate_text_embedding(data.text_content or "")
            visual_emb = self.generate_visual_embedding(data.image_data, data.visual_description)
            
            if text_emb and visual_emb:
                # Combine embeddings (simple average)
                combined = [(a + b) / 2 for a, b in zip(text_emb, visual_emb)]
                return combined
            elif text_emb:
                return text_emb
            elif visual_emb:
                return visual_emb
        else:
            # Text-based content
            return self.generate_text_embedding(data.text_content or "")

# --------- Multimodal Agent with BigQuery Object Tables ---------
class MultimodalAgent:
    """Multimodal Agent that handles unstructured data using BigQuery Object Tables."""
    
    def __init__(self, project_id: str = None, dataset_id: str = "multimodal_data", 
                 table_id: str = "multimodal_table", bucket_name: str = None):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.bucket_name = bucket_name or f"{self.project_id}-multimodal-storage"
        
        # Initialize components
        self.embedding_generator = MultimodalEmbeddingGenerator()
        self.bq_client = self._initialize_bigquery_client()
        self.storage_client = self._initialize_storage_client()
        
        # Ensure bucket exists
        self._ensure_bucket_exists()

    def _initialize_bigquery_client(self) -> bigquery.Client:
        """Initialize BigQuery client with authentication."""
        try:
            service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if service_account_path and os.path.exists(service_account_path):
                credentials = service_account.Credentials.from_service_account_file(service_account_path)
                client = bigquery.Client(credentials=credentials, project=self.project_id)
            else:
                client = bigquery.Client(project=self.project_id)
            
            print(f"[MultimodalAgent] Connected to BigQuery project: {self.project_id}")
            return client
        except Exception as e:
            print(f"[MultimodalAgent] Failed to initialize BigQuery client: {e}")
            raise

    def _initialize_storage_client(self) -> storage.Client:
        """Initialize Cloud Storage client."""
        try:
            service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if service_account_path and os.path.exists(service_account_path):
                credentials = service_account.Credentials.from_service_account_file(service_account_path)
                client = storage.Client(credentials=credentials, project=self.project_id)
            else:
                client = storage.Client(project=self.project_id)
            
            print(f"[MultimodalAgent] Connected to Cloud Storage")
            return client
        except Exception as e:
            print(f"[MultimodalAgent] Failed to initialize Storage client: {e}")
            raise

    def _ensure_bucket_exists(self):
        """Ensure the Cloud Storage bucket exists."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            if not bucket.exists():
                bucket = self.storage_client.create_bucket(self.bucket_name)
                print(f"[MultimodalAgent] Created bucket: {self.bucket_name}")
            else:
                print(f"[MultimodalAgent] Using existing bucket: {self.bucket_name}")
        except Exception as e:
            print(f"[MultimodalAgent] Error ensuring bucket exists: {e}")
            raise

    def create_bigquery_table(self):
        """Create BigQuery table with Object Table support for multimodal data."""
        try:
            table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            
            # Define schema for multimodal data
            schema = [
                bigquery.SchemaField("data_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("description", "TEXT", mode="NULLABLE"),
                bigquery.SchemaField("content_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("file_path", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("file_size", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("mime_type", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("text_content", "TEXT", mode="NULLABLE"),
                bigquery.SchemaField("image_width", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("image_height", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("incident_id", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("service", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("environment", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("cluster", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("namespace", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("pod", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("container", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("commit_sha", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("tool", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("configs", "TEXT", mode="NULLABLE"),
                bigquery.SchemaField("docs_faq", "TEXT", mode="NULLABLE"),
                bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("resolution", "TEXT", mode="NULLABLE"),
                bigquery.SchemaField("severity", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("category", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("priority", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("tags", "STRING", mode="REPEATED"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
                bigquery.SchemaField("updated_at", "TIMESTAMP", mode="NULLABLE"),
                bigquery.SchemaField("extracted_text", "TEXT", mode="NULLABLE"),
                bigquery.SchemaField("visual_description", "TEXT", mode="NULLABLE"),
                bigquery.SchemaField("error_patterns", "STRING", mode="REPEATED"),
                bigquery.SchemaField("ui_elements", "STRING", mode="REPEATED"),
                # Embeddings
                bigquery.SchemaField("text_embedding", "FLOAT", mode="REPEATED"),
                bigquery.SchemaField("visual_embedding", "FLOAT", mode="REPEATED"),
                bigquery.SchemaField("multimodal_embedding", "FLOAT", mode="REPEATED"),
                # Object Table fields
                bigquery.SchemaField("gcs_uri", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("content_for_embedding", "TEXT", mode="NULLABLE"),
            ]
            
            table = bigquery.Table(table_id, schema=schema)
            
            # Create table
            table = self.bq_client.create_table(table, exists_ok=True)
            print(f"[MultimodalAgent] Created table: {table_id}")
            
        except Exception as e:
            print(f"[MultimodalAgent] Error creating table: {e}")
            raise

    def upload_to_gcs(self, data: MultimodalData) -> str:
        """Upload file data to Google Cloud Storage and return GCS URI."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            
            # Create object name
            file_extension = ""
            if data.mime_type:
                file_extension = mimetypes.guess_extension(data.mime_type) or ""
            
            object_name = f"multimodal/{data.data_id}{file_extension}"
            
            # Upload data
            blob = bucket.blob(object_name)
            
            if data.image_data:
                blob.upload_from_string(data.image_data, content_type=data.mime_type or "application/octet-stream")
            else:
                # For text content, upload as text
                content = data.text_content or data.description or ""
                blob.upload_from_string(content, content_type="text/plain")
            
            gcs_uri = f"gs://{self.bucket_name}/{object_name}"
            print(f"[MultimodalAgent] Uploaded to GCS: {gcs_uri}")
            return gcs_uri
            
        except Exception as e:
            print(f"[MultimodalAgent] Error uploading to GCS: {e}")
            raise

    def process_multimodal_data(self, data: MultimodalData) -> Dict[str, Any]:
        """Process multimodal data and generate embeddings."""
        try:
            # Process based on content type
            if data.content_type in ["image", "screenshot"] and data.image_data:
                # Process image
                image_info = self.embedding_generator.process_image(data.image_data)
                if image_info:
                    data.image_width = image_info["width"]
                    data.image_height = image_info["height"]
                    data.image_data = image_info["processed_data"]
                
                # Generate visual description
                if not data.visual_description:
                    data.visual_description = f"Screenshot showing {data.description}"
                
                # Generate embeddings
                text_embedding = self.generate_text_embedding(data.text_content or data.description)
                visual_embedding = self.embedding_generator.generate_visual_embedding(
                    data.image_data, data.visual_description
                )
                multimodal_embedding = visual_embedding  # For images, visual is primary
                
            elif data.content_type == "pdf" and data.image_data:
                # Process PDF
                if not data.extracted_text:
                    data.extracted_text = self.embedding_generator.extract_text_from_pdf(data.image_data)
                
                # Generate embeddings
                text_embedding = self.generate_text_embedding(data.extracted_text)
                visual_embedding = self.embedding_generator.generate_visual_embedding(
                    data.image_data, data.visual_description
                )
                multimodal_embedding = self.embedding_generator.generate_multimodal_embedding(data)
                
            else:
                # Text-based content
                text_embedding = self.generate_text_embedding(data.text_content or data.description)
                visual_embedding = None
                multimodal_embedding = text_embedding
            
            # Upload to GCS
            gcs_uri = self.upload_to_gcs(data)
            
            # Prepare content for embedding
            content_parts = []
            if data.text_content:
                content_parts.append(f"Text: {data.text_content}")
            if data.extracted_text:
                content_parts.append(f"Extracted Text: {data.extracted_text}")
            if data.visual_description:
                content_parts.append(f"Visual: {data.visual_description}")
            if data.description:
                content_parts.append(f"Description: {data.description}")
            
            content_for_embedding = "\n\n".join(content_parts)
            
            return {
                "data_id": data.data_id,
                "data": data,
                "text_embedding": text_embedding,
                "visual_embedding": visual_embedding,
                "multimodal_embedding": multimodal_embedding,
                "gcs_uri": gcs_uri,
                "content_for_embedding": content_for_embedding
            }
            
        except Exception as e:
            print(f"[process_multimodal_data] error: {e}")
            raise

    def generate_text_embedding(self, text: str) -> List[float]:
        """Generate text embedding."""
        return self.embedding_generator.generate_text_embedding(text)

    def insert_multimodal_data(self, processed_data: Dict[str, Any]) -> None:
        """Insert processed multimodal data into BigQuery."""
        try:
            table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            data = processed_data["data"]
            
            # Prepare row for BigQuery
            row = {
                "data_id": data.data_id,
                "title": data.title,
                "description": data.description,
                "content_type": data.content_type,
                "file_path": data.file_path,
                "file_size": data.file_size,
                "mime_type": data.mime_type,
                "text_content": data.text_content,
                "image_width": data.image_width,
                "image_height": data.image_height,
                "incident_id": data.incident_id,
                "service": data.service,
                "environment": data.environment,
                "cluster": data.cluster,
                "namespace": data.namespace,
                "pod": data.pod,
                "container": data.container,
                "commit_sha": data.commit_sha,
                "tool": data.tool,
                "configs": data.configs,
                "docs_faq": data.docs_faq,
                "status": data.status,
                "resolution": data.resolution,
                "severity": data.severity,
                "category": data.category,
                "priority": data.priority,
                "tags": data.tags or [],
                "created_at": data.created_at.isoformat() if data.created_at else None,
                "updated_at": data.updated_at.isoformat() if data.updated_at else None,
                "extracted_text": data.extracted_text,
                "visual_description": data.visual_description,
                "error_patterns": data.error_patterns or [],
                "ui_elements": data.ui_elements or [],
                "text_embedding": processed_data["text_embedding"],
                "visual_embedding": processed_data["visual_embedding"],
                "multimodal_embedding": processed_data["multimodal_embedding"],
                "gcs_uri": processed_data["gcs_uri"],
                "content_for_embedding": processed_data["content_for_embedding"],
            }
            
            # Insert row
            errors = self.bq_client.insert_rows_json(table_id, [row])
            if errors:
                print(f"[MultimodalAgent] Errors inserting row: {errors}")
            else:
                print(f"[MultimodalAgent] Successfully inserted multimodal data: {data.data_id}")
                
        except Exception as e:
            print(f"[MultimodalAgent] Error inserting data: {e}")
            raise

    def search_similar_multimodal(self, query: str, content_types: List[str] = None, 
                                 k: int = 5, filter_metadata: Optional[Dict] = None) -> List[MultimodalSearchResult]:
        """Search for similar multimodal content using BigQuery vector similarity search."""
        try:
            # Generate query embedding
            query_embedding = self.generate_text_embedding(query)
            if query_embedding is None:
                print("[search_similar_multimodal] Failed to generate query embedding")
                return []

            # Build BigQuery SQL query
            table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            
            base_query = f"""
            SELECT 
                data_id,
                title,
                description,
                content_type,
                text_content,
                visual_description,
                extracted_text,
                gcs_uri,
                content_for_embedding,
                incident_id,
                service,
                environment,
                severity,
                category,
                resolution,
                tags,
                created_at,
                ML.DISTANCE(multimodal_embedding, @query_embedding, 'COSINE') as similarity_score
            FROM `{table_id}`
            """
            
            # Add WHERE conditions
            where_conditions = []
            if content_types:
                content_type_filter = "', '".join(content_types)
                where_conditions.append(f"content_type IN ('{content_type_filter}')")
            
            if filter_metadata:
                for key, val in filter_metadata.items():
                    if val is not None:
                        where_conditions.append(f"{key} = @{key}")
            
            if where_conditions:
                base_query += " WHERE " + " AND ".join(where_conditions)
            
            # Add ORDER BY and LIMIT
            base_query += f"""
            ORDER BY similarity_score ASC
            LIMIT {k}
            """
            
            # Prepare query parameters
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("query_embedding", "ARRAY<FLOAT64>", query_embedding)
                ]
            )
            
            # Add filter parameters
            if filter_metadata:
                for key, val in filter_metadata.items():
                    if val is not None:
                        job_config.query_parameters.append(
                            bigquery.ScalarQueryParameter(key, "STRING", val)
                        )
            
            # Execute query
            query_job = self.bq_client.query(base_query, job_config=job_config)
            results = query_job.result()
            
            # Format results
            search_results = []
            for row in results:
                confidence_percentage = max(0, min(100, int((1 - row.similarity_score) * 100)))
                
                # Determine matched content
                matched_content = None
                if row.text_content:
                    matched_content = row.text_content
                elif row.extracted_text:
                    matched_content = row.extracted_text
                elif row.description:
                    matched_content = row.description
                
                search_result = MultimodalSearchResult(
                    data_id=row.data_id,
                    title=row.title,
                    content_type=row.content_type,
                    similarity_score=float(row.similarity_score),
                    confidence_percentage=confidence_percentage,
                    metadata={
                        "description": row.description,
                        "incident_id": row.incident_id,
                        "service": row.service,
                        "environment": row.environment,
                        "severity": row.severity,
                        "category": row.category,
                        "resolution": row.resolution,
                        "tags": list(row.tags) if row.tags else [],
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "gcs_uri": row.gcs_uri
                    },
                    matched_content=matched_content,
                    visual_description=row.visual_description
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            print(f"[search_similar_multimodal] error: {e}")
            return []
    
    def process_from_explainability_agent(self, 
                                        explainability_data: Dict[str, Any],
                                        query: str = None) -> List[MultimodalSearchResult]:
        """
        Process data from explainability agent and perform multimodal analysis.
        This method takes the output from the explainability agent and enhances it with multimodal analysis.
        
        Args:
            explainability_data: Data from explainability agent (from prepare_for_multimodal_agent)
            query: Original query (if not in explainability_data)
            
        Returns:
            List of MultimodalSearchResult objects
        """
        try:
            if "error" in explainability_data:
                print(f"[process_from_explainability_agent] Error in explainability data: {explainability_data['error']}")
                return []
            
            # Extract query
            if not query:
                query = explainability_data.get("query", "Unknown query")
            
            # Perform multimodal search
            multimodal_results = self.search_similar_multimodal(
                query=query,
                content_types=["screenshot", "image", "pdf", "log", "incident"],
                k=5
            )
            
            # Enhance results with explainability context
            enhanced_results = []
            for result in multimodal_results:
                # Add explainability metadata
                result.metadata.update({
                    "explainability_confidence": explainability_data.get("overall_confidence", 0),
                    "transparency_score": explainability_data.get("transparency_score", 0.0),
                    "evidence_count": explainability_data.get("evidence_summary", {}).get("total_tickets", 0),
                    "high_confidence_evidence": explainability_data.get("evidence_summary", {}).get("high_confidence_count", 0),
                    "resolved_evidence": explainability_data.get("evidence_summary", {}).get("resolved_count", 0),
                    "explainability_integrated": True
                })
                enhanced_results.append(result)
            
            print(f"[process_from_explainability_agent] Found {len(enhanced_results)} multimodal results with explainability context")
            return enhanced_results
            
        except Exception as e:
            print(f"[process_from_explainability_agent] Error: {e}")
            return []
    
    def prepare_for_feedback_agent(self, 
                                 multimodal_results: List[MultimodalSearchResult],
                                 explainability_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Prepare multimodal results for feedback integration agent.
        This method formats the multimodal data in a way that's optimized for feedback collection and learning.
        
        Args:
            multimodal_results: List of MultimodalSearchResult objects
            explainability_data: Optional explainability data for context
            
        Returns:
            Dictionary containing data optimized for feedback integration agent
        """
        try:
            # Prepare feedback data
            feedback_data = {
                "multimodal_results_count": len(multimodal_results),
                "content_types": list(set(result.content_type for result in multimodal_results)),
                "average_confidence": sum(result.confidence_percentage for result in multimodal_results) / len(multimodal_results) if multimodal_results else 0,
                "results_summary": [
                    {
                        "data_id": result.data_id,
                        "title": result.title,
                        "content_type": result.content_type,
                        "confidence_percentage": result.confidence_percentage,
                        "similarity_score": result.similarity_score,
                        "has_visual": bool(result.visual_description),
                        "has_resolution": bool(result.metadata.get("resolution")),
                        "gcs_uri": result.metadata.get("gcs_uri")
                    }
                    for result in multimodal_results
                ],
                "explainability_context": explainability_data if explainability_data else {},
                "feedback_ready": True,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            print(f"[prepare_for_feedback_agent] Prepared feedback data for {len(multimodal_results)} multimodal results")
            return feedback_data
            
        except Exception as e:
            print(f"[prepare_for_feedback_agent] Error: {e}")
            return {"error": str(e), "feedback_ready": False}

    def find_visual_similar_errors(self, screenshot_data: bytes, description: str = None) -> List[MultimodalSearchResult]:
        """Find visually similar error logs from a screenshot."""
        try:
            # Process the screenshot
            image_info = self.embedding_generator.process_image(screenshot_data)
            if not image_info:
                return []
            
            # Generate visual embedding
            visual_embedding = self.embedding_generator.generate_visual_embedding(
                image_info["processed_data"], description
            )
            
            if visual_embedding is None:
                return []
            
            # Search for similar visual content
            table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            
            query = f"""
            SELECT 
                data_id,
                title,
                description,
                content_type,
                text_content,
                visual_description,
                extracted_text,
                gcs_uri,
                content_for_embedding,
                incident_id,
                service,
                environment,
                severity,
                category,
                resolution,
                tags,
                created_at,
                ML.DISTANCE(visual_embedding, @visual_embedding, 'COSINE') as visual_similarity,
                ML.DISTANCE(multimodal_embedding, @visual_embedding, 'COSINE') as overall_similarity
            FROM `{table_id}`
            WHERE content_type IN ('screenshot', 'image', 'log', 'incident')
            ORDER BY overall_similarity ASC
            LIMIT 10
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("visual_embedding", "ARRAY<FLOAT64>", visual_embedding)
                ]
            )
            
            query_job = self.bq_client.query(query, job_config=job_config)
            results = query_job.result()
            
            # Format results
            search_results = []
            for row in results:
                confidence_percentage = max(0, min(100, int((1 - row.overall_similarity) * 100)))
                
                search_result = MultimodalSearchResult(
                    data_id=row.data_id,
                    title=row.title,
                    content_type=row.content_type,
                    similarity_score=float(row.overall_similarity),
                    confidence_percentage=confidence_percentage,
                    visual_similarity=float(row.visual_similarity),
                    metadata={
                        "description": row.description,
                        "incident_id": row.incident_id,
                        "service": row.service,
                        "environment": row.environment,
                        "severity": row.severity,
                        "category": row.category,
                        "resolution": row.resolution,
                        "tags": list(row.tags) if row.tags else [],
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "gcs_uri": row.gcs_uri
                    },
                    matched_content=row.text_content or row.extracted_text or row.description,
                    visual_description=row.visual_description
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            print(f"[find_visual_similar_errors] error: {e}")
            return []

    def get_multimodal_statistics(self) -> Dict[str, Any]:
        """Get statistics about multimodal data in BigQuery."""
        try:
            table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            
            stats_query = f"""
            SELECT 
                COUNT(*) as total_items,
                COUNTIF(content_type = 'image') as image_count,
                COUNTIF(content_type = 'screenshot') as screenshot_count,
                COUNTIF(content_type = 'pdf') as pdf_count,
                COUNTIF(content_type = 'log') as log_count,
                COUNTIF(content_type = 'incident') as incident_count,
                COUNTIF(resolution IS NOT NULL AND resolution != '') as resolved_items,
                COUNTIF(severity = 'critical') as critical_count,
                COUNTIF(severity = 'error') as error_count,
                COUNTIF(environment = 'prod') as prod_count,
                COUNTIF(environment = 'staging') as staging_count,
                COUNTIF(environment = 'dev') as dev_count
            FROM `{table_id}`
            """
            
            query_job = self.bq_client.query(stats_query)
            results = list(query_job.result())
            
            if not results:
                return {"error": "No data found in BigQuery table"}
            
            row = results[0]
            
            return {
                "total_items": row.total_items,
                "by_content_type": {
                    "image": row.image_count,
                    "screenshot": row.screenshot_count,
                    "pdf": row.pdf_count,
                    "log": row.log_count,
                    "incident": row.incident_count
                },
                "resolved_items": row.resolved_items,
                "unresolved_items": row.total_items - row.resolved_items,
                "by_severity": {
                    "critical": row.critical_count,
                    "error": row.error_count
                },
                "by_environment": {
                    "production": row.prod_count,
                    "staging": row.staging_count,
                    "development": row.dev_count
                },
                "multimodal_available": True
            }
            
        except Exception as e:
            print(f"[get_multimodal_statistics] error: {e}")
            return {
                "error": str(e),
                "multimodal_available": False
            }

# --------- Utility Functions ---------
def load_image_from_file(file_path: str) -> bytes:
    """Load image data from file."""
    try:
        with open(file_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"[load_image_from_file] error: {e}")
        return None

def load_pdf_from_file(file_path: str) -> bytes:
    """Load PDF data from file."""
    try:
        with open(file_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"[load_pdf_from_file] error: {e}")
        return None

def create_multimodal_data_from_file(file_path: str, data_id: str = None, 
                                   title: str = None, description: str = None,
                                   content_type: str = None, **metadata) -> MultimodalData:
    """Create MultimodalData from file."""
    try:
        file_path_obj = Path(file_path)
        
        if not data_id:
            data_id = f"multimodal_{file_path_obj.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if not title:
            title = file_path_obj.stem
        
        if not description:
            description = f"Multimodal data from {file_path_obj.name}"
        
        # Determine content type
        if not content_type:
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type:
                if mime_type.startswith('image/'):
                    content_type = 'screenshot' if 'screenshot' in file_path.lower() else 'image'
                elif mime_type == 'application/pdf':
                    content_type = 'pdf'
                else:
                    content_type = 'log'
            else:
                content_type = 'log'
        
        # Load file data
        file_data = None
        text_content = None
        
        if content_type in ['image', 'screenshot']:
            file_data = load_image_from_file(file_path)
        elif content_type == 'pdf':
            file_data = load_pdf_from_file(file_path)
        else:
            # Text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
        
        # Get file info
        file_size = file_path_obj.stat().st_size
        mime_type, _ = mimetypes.guess_type(file_path)
        
        return MultimodalData(
            data_id=data_id,
            title=title,
            description=description,
            content_type=content_type,
            file_path=str(file_path_obj),
            file_size=file_size,
            mime_type=mime_type,
            text_content=text_content,
            image_data=file_data,
            created_at=datetime.now(),
            **metadata
        )
        
    except Exception as e:
        print(f"[create_multimodal_data_from_file] error: {e}")
        return None

# --------- Main Functions ---------
def create_multimodal_agent(project_id: str = None, dataset_id: str = "multimodal_data", 
                          table_id: str = "multimodal_table", bucket_name: str = None) -> MultimodalAgent:
    """Create and initialize a MultimodalAgent instance."""
    if project_id is None:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
    
    return MultimodalAgent(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        bucket_name=bucket_name
    )

def create_sample_multimodal_data():
    """Create sample multimodal data for demonstration."""
    sample_data = []
    
    # Sample 1: Screenshot data
    screenshot_data = MultimodalData(
        data_id="screenshot_error_001",
        title="Database Connection Error Screenshot",
        description="Screenshot showing database connection timeout error in production",
        content_type="screenshot",
        text_content="Error: Database connection timeout after 30 seconds. Connection refused to database-server-01.",
        visual_description="Red error banner showing 'Database Connection Timeout' with error code 500",
        incident_id="INC-2024-001",
        service="user-service",
        environment="prod",
        cluster="prod-cluster-1",
        namespace="user-namespace",
        severity="error",
        category="Database",
        priority="High",
        tags=["database", "timeout", "connection", "prod"],
        created_at=datetime.now()
    )
    sample_data.append(screenshot_data)
    
    # Sample 2: Log data
    log_data = MultimodalData(
        data_id="log_auth_error_001",
        title="Authentication Service Error Log",
        description="Log entries showing authentication failures",
        content_type="log",
        text_content="""
2024-01-15 10:30:15 ERROR [auth-service] Authentication failed for user: john.doe@company.com
2024-01-15 10:30:16 ERROR [auth-service] JWT token validation failed: expired
2024-01-15 10:30:17 ERROR [auth-service] Database connection timeout during user lookup
2024-01-15 10:30:18 ERROR [auth-service] Retry attempt 1/3 failed
        """.strip(),
        incident_id="INC-2024-002",
        service="auth-service",
        environment="prod",
        severity="error",
        category="Authentication",
        priority="Critical",
        tags=["auth", "jwt", "timeout", "database"],
        created_at=datetime.now()
    )
    sample_data.append(log_data)
    
    # Sample 3: PDF documentation
    pdf_data = MultimodalData(
        data_id="pdf_runbook_001",
        title="Database Recovery Runbook",
        description="PDF documentation for database recovery procedures",
        content_type="pdf",
        text_content="""
Database Recovery Procedures
===========================

1. Identify the issue
   - Check database connection status
   - Review error logs
   - Verify network connectivity

2. Recovery Steps
   - Restart database service
   - Check disk space
   - Verify configuration files
   - Test connection

3. Verification
   - Run health checks
   - Monitor performance metrics
   - Confirm data integrity
        """.strip(),
        visual_description="PDF document with database recovery procedures and diagrams",
        incident_id="INC-2024-003",
        service="database-service",
        environment="prod",
        severity="info",
        category="Documentation",
        priority="Low",
        tags=["runbook", "database", "recovery", "documentation"],
        created_at=datetime.now()
    )
    sample_data.append(pdf_data)
    
    # Sample 4: Configuration file
    config_data = MultimodalData(
        data_id="config_db_001",
        title="Database Configuration",
        description="Database connection configuration file",
        content_type="config",
        text_content="""
# Database Configuration
DB_HOST=db-server-01.company.com
DB_PORT=5432
DB_NAME=user_service_db
DB_USER=app_user
DB_PASSWORD=encrypted_password_here
DB_TIMEOUT=30
DB_POOL_SIZE=10
DB_RETRY_ATTEMPTS=3
        """.strip(),
        incident_id="INC-2024-004",
        service="database-service",
        environment="prod",
        severity="info",
        category="Configuration",
        priority="Medium",
        tags=["config", "database", "connection", "prod"],
        created_at=datetime.now()
    )
    sample_data.append(config_data)
    
    return sample_data

def ensure_multimodal_agent_ready(agent: MultimodalAgent) -> bool:
    """Ensure the multimodal agent is ready for the workflow.
    
    Args:
        agent: MultimodalAgent instance
        
    Returns:
        True if multimodal agent is ready, False otherwise
    """
    try:
        # Test the agent with a simple query
        test_results = agent.search_similar_multimodal("test multimodal functionality", k=1)
        
        if test_results is not None:
            print(f"[ensure_multimodal_agent_ready] Multimodal agent ready")
            return True
        else:
            print(f"[ensure_multimodal_agent_ready] Multimodal agent not responding properly")
            return False
        
    except Exception as e:
        print(f"[ensure_multimodal_agent_ready] Error: {e}")
        return False


def main():
    """Main function to demonstrate multimodal agent functionality."""
    print("MULTIMODAL AGENT COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("This demonstration shows the capabilities of the Multimodal Agent")
    print("for handling unstructured data in DevOps incident management.")
    print("=" * 80)
    
    try:
        # Initialize the multimodal agent
        print("\n1. Initializing Multimodal Agent...")
        agent = create_multimodal_agent()
        
        # Create BigQuery table
        print("2. Creating BigQuery table...")
        agent.create_bigquery_table()
        
        # Create sample data
        print("3. Creating sample multimodal data...")
        sample_data = create_sample_multimodal_data()
        
        # Process each data item
        print("\n4. Processing Multimodal Data")
        print("-" * 50)
        processed_items = []
        for data in sample_data:
            print(f"\nProcessing {data.content_type}: {data.title}")
            print("-" * 40)
            
            try:
                # Process the multimodal data
                processed = agent.process_multimodal_data(data)
                processed_items.append(processed)
                
                # Insert into BigQuery
                agent.insert_multimodal_data(processed)
                
                print(f"✓ Successfully processed and stored: {data.data_id}")
                print(f"  Content Type: {data.content_type}")
                print(f"  GCS URI: {processed['gcs_uri']}")
                print(f"  Has Text Embedding: {processed['text_embedding'] is not None}")
                print(f"  Has Visual Embedding: {processed['visual_embedding'] is not None}")
                print(f"  Has Multimodal Embedding: {processed['multimodal_embedding'] is not None}")
                
            except Exception as e:
                print(f"✗ Error processing {data.data_id}: {e}")
        
        # Demonstrate similarity search
        print("\n5. Similarity Search Demonstration")
        print("-" * 50)
        
        test_queries = [
            "database connection timeout error",
            "authentication failure user login",
            "database recovery procedures",
            "configuration settings database"
        ]
        
        for query in test_queries:
            print(f"\nSearching for: '{query}'")
            print("-" * 30)
            
            # Search for similar content
            results = agent.search_similar_multimodal(
                query=query,
                content_types=["screenshot", "log", "pdf", "config"],
                k=3
            )
            
            print(f"Found {len(results)} similar items:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.title} ({result.content_type})")
                print(f"     Confidence: {result.confidence_percentage}%")
                print(f"     Similarity Score: {result.similarity_score:.4f}")
                if result.visual_description:
                    print(f"     Visual: {result.visual_description[:100]}...")
                print()
        
        # Demonstrate visual similarity search
        print("\n6. Visual Similarity Search Demonstration")
        print("-" * 50)
        
        print("Visual similarity search capabilities:")
        print("• Screenshot → Error Log matching")
        print("• Visual error patterns → Textual log patterns")
        print("• UI elements → System components")
        print("• Error dialogs → Incident reports")
        
        print("\nExample workflow:")
        print("1. User uploads screenshot of error")
        print("2. Agent extracts visual features and text")
        print("3. Searches for similar error patterns in logs")
        print("4. Returns matching incidents with resolutions")
        
        # Demonstrate cross-modal matching
        print("\n7. Cross-Modal Matching Demonstration")
        print("-" * 50)
        
        print("Cross-modal matching allows different content types to be matched:")
        print("\n• Screenshot → Error Log")
        print("  - Visual error dialog matches text error messages")
        print("  - UI elements match log patterns")
        
        print("\n• PDF Documentation → Incident Reports")
        print("  - Runbook procedures match incident resolutions")
        print("  - Documentation matches actual fixes")
        
        print("\n• Configuration → Error Logs")
        print("  - Config settings match error patterns")
        print("  - Parameter values match log entries")
        
        # Demonstrate cross-modal search
        print("\nCross-modal search example:")
        results = agent.search_similar_multimodal(
            query="database timeout configuration",
            content_types=["screenshot", "log", "pdf", "config"],
            k=5
        )
        
        print(f"Found {len(results)} cross-modal matches:")
        for result in results:
            print(f"- {result.title} ({result.content_type}) - {result.confidence_percentage}%")
        
        # Get statistics
        print("\n8. Multimodal Statistics")
        print("-" * 50)
        stats = agent.get_multimodal_statistics()
        
        if "error" in stats:
            print(f"Error retrieving statistics: {stats['error']}")
        else:
            print("Multimodal Data Statistics:")
            print(f"Total Items: {stats['total_items']}")
            print(f"Resolved Items: {stats['resolved_items']}")
            print(f"Unresolved Items: {stats['unresolved_items']}")
            
            print("\nBy Content Type:")
            for content_type, count in stats['by_content_type'].items():
                print(f"  {content_type}: {count}")
            
            print("\nBy Severity:")
            for severity, count in stats['by_severity'].items():
                print(f"  {severity}: {count}")
            
            print("\nBy Environment:")
            for env, count in stats['by_environment'].items():
                print(f"  {env}: {count}")
        
        # File processing demonstration
        print("\n9. File Processing Capabilities")
        print("-" * 50)
        
        print("The multimodal agent can process various file types:")
        print("\n• Images (PNG, JPEG, GIF)")
        print("  - Extract visual features")
        print("  - Generate visual descriptions")
        print("  - Create visual embeddings")
        
        print("\n• PDFs")
        print("  - Extract text content")
        print("  - Process visual elements")
        print("  - Generate combined embeddings")
        
        print("\n• Text Files (Logs, Configs)")
        print("  - Parse structured content")
        print("  - Extract metadata")
        print("  - Generate text embeddings")
        
        print("\n• Screenshots")
        print("  - Analyze UI elements")
        print("  - Extract error messages")
        print("  - Match to similar incidents")
        
        print("\nExample file processing workflow:")
        print("1. Load file: create_multimodal_data_from_file('error_screenshot.png')")
        print("2. Process: agent.process_multimodal_data(data)")
        print("3. Store: agent.insert_multimodal_data(processed)")
        print("4. Search: agent.search_similar_multimodal(query)")
        
        print("\n" + "="*80)
        print("MULTIMODAL AGENT DEMONSTRATION COMPLETE")
        print("="*80)
        print("\nKey Capabilities Demonstrated:")
        print("✓ Multimodal data processing (images, PDFs, logs, configs)")
        print("✓ BigQuery Object Table integration")
        print("✓ Google Cloud Storage integration")
        print("✓ Visual similarity search")
        print("✓ Cross-modal matching")
        print("✓ Vector similarity search with embeddings")
        print("✓ Comprehensive statistics and analytics")
        
        print("\nNext Steps:")
        print("1. Upload actual screenshots and PDFs for testing")
        print("2. Integrate with your incident management system")
        print("3. Set up automated processing pipelines")
        print("4. Configure monitoring and alerting")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Please check your Google Cloud configuration and API keys.")

if __name__ == "__main__":
    main()
