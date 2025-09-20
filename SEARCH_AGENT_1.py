"""
Search Agent - Vector Search Implementation using LangChain and BigFrames
Performs semantic similarity search on tickets with metadata retrieval
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Literal, Union
from dataclasses import dataclass 
 

from datetime import datetime

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.schema import Document
from langchain_google_community import BigQueryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Google Cloud and BigFrames imports
import bigframes
from google.cloud import bigquery
from google.oauth2 import service_account
import google.generativeai as genai

# Vector similarity and search
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


@dataclass
class SearchResult:
    """Data class for search results"""
    log_id: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    related_context: Optional[Dict[str, Any]] = None

@dataclass
class LogData:
    log_id: str
    title: str
    description: str
    # What artifact this is: log lines, config text, documentation, or incident report
    source_type: Literal["log", "config", "doc", "incident"] = "log"
    # Optional linking/context
    incident_id: Optional[str] = None
    service: Optional[str] = None
    environment: Optional[str] = None  # dev, staging, prod
    cluster: Optional[str] = None
    namespace: Optional[str] = None
    pod: Optional[str] = None
    container: Optional[str] = None
    file_path: Optional[str] = None  # for configs/docs
    commit_sha: Optional[str] = None  # CI/CD association
    tool: Optional[str] = None  # e.g., github-actions, argo, jenkins
    # Domain content
    configs: Optional[str] = None
    docs_faq: Optional[str] = None
    status: Optional[str] = None
    resolution: Optional[str] = None
    severity: Optional[str] = None  # info, warn, error, critical
    category: Optional[str] = None  # Authentication, Database, etc.
    priority: Optional[str] = None  # Low/Medium/High/Critical
    tags: Optional[List[str]] = None
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class VectorSearchAgent:
    """
    Search Agent that performs vector search on tickets using LangChain and BigFrames
    """
    
    def __init__(self, 
                 project_id: str = "big-station-472112-i1",
                 credentials_path: str = "big-station-472112-i1-01b16573569e.json",
                 api_key_path: str = "Gemini_API_Key.txt"):
        """
        Initialize the VectorSearchAgent
        
        Args:
            project_id: Google Cloud project ID
            credentials_path: Path to service account credentials
            api_key_path: Path to Gemini API key
        """
        self.project_id = project_id
        self.credentials_path = credentials_path
        self.api_key_path = api_key_path
        
        # Initialize components
        self._setup_authentication()
        self._setup_embeddings()
        self._setup_bigframes()
        self._setup_bigquery()
        
        # Initialize vector store
        self.vector_store = None
        # Text splitter for larger artifacts (configs/docs/incidents)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", ".", " "]
        )

    def _artifact_to_text(self, item: Union[LogData, Dict[str, Any]]) -> str:
        """Build a unified textual representation for embeddings from log/config/doc/incident."""
        if isinstance(item, dict):
            data = item
        else:
            data = item.__dict__
        parts: List[str] = []
        title = data.get("title")
        description = data.get("description")
        configs = data.get("configs")
        docs_faq = data.get("docs_faq")
        source_type = data.get("source_type", "log")
        if title:
            parts.append(f"Title: {title}")
        if source_type:
            parts.append(f"Type: {source_type}")
        if description:
            parts.append(f"Description: {description}")
        if configs:
            parts.append(f"Config:\n{configs}")
        if docs_faq:
            parts.append(f"Docs/FAQ:\n{docs_faq}")
        # Include light-weight metadata signals
        meta_keys = ["service", "environment", "cluster", "namespace", "pod", "container", "incident_id", "commit_sha", "tool", "severity", "category", "priority"]
        meta_kv = [f"{k}: {data.get(k)}" for k in meta_keys if data.get(k)]
        if meta_kv:
            parts.append("Context:\n" + "\n".join(meta_kv))
        return "\n\n".join(parts).strip()

    def _artifact_to_metadata(self, item: Union[LogData, Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize metadata for storage in vector DB - using minimal fields to match existing schema."""
        if isinstance(item, dict):
            data = item
        else:
            data = item.__dict__
        
        # Use only basic fields that are likely to exist in the table
        metadata: Dict[str, Any] = {
            "log_id": data.get("log_id", ""),
            "priority": data.get("priority", "Low"),
            "status": data.get("status", "open"),
        }
        
        # Handle timestamps
        created_at = data.get("created_at")
        if isinstance(created_at, datetime):
            metadata["created_at"] = created_at.isoformat()
        elif created_at:
            metadata["created_at"] = str(created_at)
        else:
            metadata["created_at"] = datetime.now().isoformat()
            
        return {k: v for k, v in metadata.items() if v is not None}
        
    def _setup_authentication(self):
        """Set up Google Cloud authentication"""
        try:
            # Set up service account credentials
            self.credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            
            # Set environment variable for BigFrames
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
            
            print("✓ Authentication configured successfully")
            
        except Exception as e:
            print(f"✗ Authentication setup failed: {e}")
            raise
    
    def _setup_embeddings(self):
        """Set up Gemini embeddings model"""
        try:
            # Read API key
            with open(self.api_key_path, 'r') as f:
                api_key = f.read().strip()
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Initialize embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
            
            print("✓ Embeddings model configured successfully")
            
        except Exception as e:
            print(f"✗ Embeddings setup failed: {e}")
            raise
    
    def _setup_bigframes(self):
        """Set up BigFrames session if available; otherwise fall back to BigQuery client."""
        try:
            # Prefer modern import path if present
            try:
                import bigframes.pandas as bpd  # type: ignore
                # Some versions expose connect under top-level, others under pandas
                if hasattr(bpd, "connect"):
                    self.bigframes_session = bpd.connect(
                        project_id=self.project_id,
                        credentials=self.credentials
                    )
                else:
                    self.bigframes_session = None
            except Exception:
                # Fallback: not available
                self.bigframes_session = None

            if self.bigframes_session is not None:
                print("✓ BigFrames session configured successfully")
            else:
                print("! BigFrames not available; will use BigQuery client for data access")

        except Exception as e:
            print(f"! BigFrames setup encountered an issue, using BigQuery client instead: {e}")
            self.bigframes_session = None
    
    def _setup_bigquery(self):
        """Set up BigQuery client"""
        try:
            self.bq_client = bigquery.Client(
                project=self.project_id,
                credentials=self.credentials
            )
            
            print("✓ BigQuery client configured successfully")
            
        except Exception as e:
            print(f"✗ BigQuery setup failed: {e}")
            raise
    
    def create_vector_store(self, 
                           dataset_id: str = "log_dataset",
                           table_id: str = "log_vectors",
                           text_column: str = "artifact_text",
                           metadata_columns: List[str] = None,
                           force_recreate: bool = False):
        """
        Create or connect to a vector store in BigQuery
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID for vectors
            text_column: Column containing text content
            metadata_columns: Additional metadata columns to include
            force_recreate: If True, drop and recreate the table
        """
        try:
            if metadata_columns is None:
                # Use minimal metadata fields that are likely to exist in the table
                metadata_columns = [
                    "log_id", "created_at", "priority", "status"
                ]
            
            # If force_recreate is True, drop the existing table
            if force_recreate:
                try:
                    table_ref = self.bq_client.dataset(dataset_id).table(table_id)
                    self.bq_client.delete_table(table_ref)
                    print(f"✓ Dropped existing table: {dataset_id}.{table_id}")
                except Exception as e:
                    print(f"! Could not drop table (may not exist): {e}")
            
            # Create vector store
            self.vector_store = BigQueryVectorStore(
                project_id=self.project_id,
                dataset_name=dataset_id,
                table_name=table_id,
                location="US",
                embedding=self.embeddings,
                content_field=text_column,
                metadata_fields=metadata_columns
            )
            
            print(f"✓ Vector store created/connected: {dataset_id}.{table_id}")
            
        except Exception as e:
            print(f"✗ Vector store creation failed: {e}")
            print("Try running with force_recreate=True to drop and recreate the table")
            raise
    
    def add_logs_to_vector_store(self, logs_data: List[Union[Dict[str, Any], LogData]]):
        """
        Ingest logs/configs/docs/incidents into the vector store
        
        Args:
            logs_data: List of log dictionaries with content and metadata
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized. Call create_vector_store() first.")
            
            # Convert to chunked documents with normalized metadata
            documents: List[Document] = []
            for item in logs_data:
                full_text = self._artifact_to_text(item)
                meta = self._artifact_to_metadata(item)
                chunks = self.text_splitter.split_text(full_text) if full_text else [""]
                for idx, chunk in enumerate(chunks):
                    chunk_meta = dict(meta)
                    chunk_meta["chunk_index"] = idx
                    doc = Document(page_content=chunk, metadata=chunk_meta)
                    documents.append(doc)
            
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            
            print(f"✓ Added {len(documents)} chunks from {len(logs_data)} artifacts to vector store")
            
        except Exception as e:
            print(f"✗ Failed to add logs to vector store: {e}")
            raise
    
    def vector_search(self, 
                     query: str, 
                     k: int = 5,
                     filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Perform vector search to find top-K semantically similar logs
        
        Args:
            query: Search query string
            k: Number of top results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of SearchResult objects with similarity scores and metadata
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized. Call create_vector_store() first.")
            
            # Perform similarity search
            if filter_metadata:
                docs_with_scores = self.vector_store.similarity_search_with_score(
                    query, k=k, filter=filter_metadata
                )
            else:
                docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Convert to SearchResult objects and enrich with RCA context
            results = []
            for doc, score in docs_with_scores:
                base = SearchResult(
                    log_id=doc.metadata.get('log_id', ''),
                    content=doc.page_content,
                    similarity_score=float(score),
                    metadata=doc.metadata,
                )
                base.related_context = self._fetch_related_context(doc.metadata)
                results.append(base)
            
            print(f"✓ Found {len(results)} similar logs for query: '{query}'")
            return results
            
        except Exception as e:
            print(f"✗ Vector search failed: {e}")
            raise
    
    def search_for_resolution_agent(self, 
                                  query: str, 
                                  k: int = 5,
                                  filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search specifically optimized for resolution agent input.
        Focuses on finding incidents with resolutions and high-quality metadata.
        
        Args:
            query: Search query string
            k: Number of top results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of SearchResult objects optimized for resolution generation
        """
        try:
            # First, try to find incidents with resolutions
            resolution_results = self.vector_search(
                query=query,
                k=k*2,  # Get more results to filter
                filter_metadata=filter_metadata
            )
            
            # Filter and prioritize results with resolutions
            prioritized_results = []
            resolution_results_only = []
            
            for result in resolution_results:
                # Check if this result has resolution information
                has_resolution = (
                    result.metadata.get('resolution') or 
                    result.metadata.get('has_resolution') or
                    'resolution' in result.content.lower() or
                    'fix' in result.content.lower() or
                    'solution' in result.content.lower()
                )
                
                if has_resolution:
                    resolution_results_only.append(result)
                else:
                    prioritized_results.append(result)
            
            # Combine: resolution results first, then others
            final_results = resolution_results_only + prioritized_results
            
            # Limit to requested number
            final_results = final_results[:k]
            
            print(f"✓ Found {len(resolution_results_only)} results with resolutions out of {len(final_results)} total")
            return final_results
            
        except Exception as e:
            print(f"✗ Resolution-optimized search failed: {e}")
            # Fallback to regular search
            return self.vector_search(query, k, filter_metadata)
    def batch_vector_search(self, 
                          queries: List[str], 
                          k: int = 5) -> Dict[str, List[SearchResult]]:
        """
        Perform batch vector search for multiple queries
        
        Args:
            queries: List of search query strings
            k: Number of top results per query
            
        Returns:
            Dictionary mapping queries to their search results
        """
        try:
            results = {}
            for query in queries:
                results[query] = self.vector_search(query, k=k)
            
            print(f"✓ Completed batch search for {len(queries)} queries")
            return results
            
        except Exception as e:
            print(f"✗ Batch vector search failed: {e}")
            raise
    
    def get_log_metadata(self, log_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a specific log using BigFrames
        
        Args:
            log_id: ID of the log to retrieve
            
        Returns:
            Dictionary containing log metadata or None if not found
        """
        try:
            # Prefer BigFrames if session supports read_gbq
            if self.bigframes_session is not None and hasattr(self.bigframes_session, "read_gbq"):
                query = f"""
                SELECT *
                FROM `{self.project_id}.log_dataset.log_metadata`
                WHERE log_id = @log_id
                """
                df = self.bigframes_session.read_gbq(
                    query,
                    parameters={"log_id": log_id},
                )
                if len(df) > 0:
                    return df.iloc[0].to_dict()
                return None
            # Fallback to BigQuery client
            query = f"""
            SELECT *
            FROM `{self.project_id}.log_dataset.log_metadata`
            WHERE log_id = @log_id
            LIMIT 1
            """
            job = self.bq_client.query(
                query,
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ScalarQueryParameter("log_id", "STRING", log_id)]
                ),
            )
            rows = list(job.result())
            if rows:
                return dict(rows[0])
            return None
        except Exception as e:
            print(f"✗ Failed to retrieve metadata for log {log_id}: {e}")
            return None

    def _fetch_related_context(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch related artifacts for RCA using available metadata keys via BigFrames."""
        context: Dict[str, Any] = {}
        try:
            service = metadata.get("service")
            environment = metadata.get("environment")
            pod = metadata.get("pod")
            incident_id = metadata.get("incident_id")
            commit_sha = metadata.get("commit_sha")

            # Recent logs for same service/env
            if service and environment:
                q_logs = f"""
                SELECT log_id, title, severity, status, created_at
                FROM `{self.project_id}.log_dataset.logs`
                WHERE service = @service AND environment = @env
                ORDER BY created_at DESC
                LIMIT 5
                """
                job = self.bq_client.query(q_logs, job_config=bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("service", "STRING", service),
                        bigquery.ScalarQueryParameter("env", "STRING", environment),
                    ]
                ))
                context["related_logs"] = [dict(row) for row in job.result()]

            # Current pod neighbors
            if pod:
                q_pod = f"""
                SELECT log_id, severity, title, created_at
                FROM `{self.project_id}.log_dataset.logs`
                WHERE pod = @pod
                ORDER BY created_at DESC
                LIMIT 5
                """
                job = self.bq_client.query(q_pod, job_config=bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ScalarQueryParameter("pod", "STRING", pod)]
                ))
                context["pod_logs"] = [dict(row) for row in job.result()]

            # Incident context
            if incident_id:
                q_inc = f"""
                SELECT incident_id, title, status, resolution, severity, created_at
                FROM `{self.project_id}.log_dataset.incidents`
                WHERE incident_id = @iid
                LIMIT 1
                """
                job = self.bq_client.query(q_inc, job_config=bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ScalarQueryParameter("iid", "STRING", incident_id)]
                ))
                context["incident"] = [dict(row) for row in job.result()]

            # Config for service
            if service:
                q_cfg = f"""
                SELECT file_path, updated_at, snippet
                FROM `{self.project_id}.log_dataset.configs`
                WHERE service = @service
                ORDER BY updated_at DESC
                LIMIT 3
                """
                job = self.bq_client.query(q_cfg, job_config=bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ScalarQueryParameter("service", "STRING", service)]
                ))
                context["configs"] = [dict(row) for row in job.result()]

            # CI/CD deployments by commit
            if commit_sha:
                q_deploy = f"""
                SELECT commit_sha, tool, status, deployed_at, env
                FROM `{self.project_id}.log_dataset.deployments`
                WHERE commit_sha = @sha
                ORDER BY deployed_at DESC
                LIMIT 3
                """
                job = self.bq_client.query(q_deploy, job_config=bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ScalarQueryParameter("sha", "STRING", commit_sha)]
                ))
                context["deployments"] = [dict(row) for row in job.result()]

        except Exception as e:
            context["_error"] = str(e)
        return context
    
    def analyze_search_results(self, results: List[SearchResult]) -> Dict[str, Any]:
        """
        Analyze search results and provide insights
        
        Args:
            results: List of SearchResult objects
            
        Returns:
            Dictionary containing analysis insights
        """
        try:
            if not results:
                return {"message": "No results to analyze"}
            
            # Calculate statistics
            scores = [r.similarity_score for r in results]
            
            analysis = {
                "total_results": len(results),
                "avg_similarity_score": np.mean(scores),
                "max_similarity_score": np.max(scores),
                "min_similarity_score": np.min(scores),
                "score_std": np.std(scores),
                "priority_distribution": {},
                "status_distribution": {}
            }
            
            # Analyze metadata distributions
            for result in results:
                priority = result.metadata.get('priority', 'unknown')
                status = result.metadata.get('status', 'unknown')
                
                analysis["priority_distribution"][priority] = \
                    analysis["priority_distribution"].get(priority, 0) + 1
                analysis["status_distribution"][status] = \
                    analysis["status_distribution"].get(status, 0) + 1
            
            return analysis
            
        except Exception as e:
            print(f"✗ Analysis failed: {e}")
            return {"error": str(e)}


    def ensure_search_agent_ready(agent: 'VectorSearchAgent', 
                                force_recreate: bool = False) -> bool:
        """
        Ensure the search agent is ready for the workflow.

        Args:
            agent: VectorSearchAgent instance
            force_recreate: Whether to force recreate the vector store

        Returns:
            True if search agent is ready, False otherwise
        """
        try:
            # Create or connect to vector store
            agent.create_vector_store(force_recreate=force_recreate)

            # Test search functionality
            test_query = "test search functionality"
            test_results = agent.vector_search(test_query, k=1)

            print(f"[ensure_search_agent_ready] Search agent ready with {len(test_results)} test results")
            return True

        except Exception as e:
            print(f"[ensure_search_agent_ready] Error: {e}")
            return False
    def main():
        """
        Main function demonstrating the VectorSearchAgent usage
        """
        try:
            # Initialize the search agent
            print("🚀 Initializing Vector Search Agent...")
            agent = VectorSearchAgent()
            
            # Create vector store (force recreate to fix schema mismatch)
            print("\n📊 Setting up vector store...")
            agent.create_vector_store(force_recreate=True)
            
            # Load incident data from JSON file
            print("\n📂 Loading incident data from JSON file...")
            try:
                with open('incident_dataset.json', 'r') as f:
                    json_data = json.load(f)
                
                # Convert JSON data to LogData objects (limit to first 100 for demo)
                sample_logs = []
                for item in json_data[:100]:  # Limit to first 100 records for demo
                    # Parse datetime strings if they exist
                    created_at = None
                    updated_at = None
                    if item.get('created_at'):
                        try:
                            created_at = datetime.fromisoformat(item['created_at'].replace('Z', '+00:00'))
                        except:
                            created_at = datetime.now()
                    if item.get('updated_at'):
                        try:
                            updated_at = datetime.fromisoformat(item['updated_at'].replace('Z', '+00:00'))
                        except:
                            updated_at = datetime.now()
                    
                    log_data = LogData(
                        log_id=item.get('log_id', ''),
                        title=item.get('title', ''),
                        description=item.get('description', ''),
                        source_type=item.get('source_type', 'log'),
                        incident_id=item.get('incident_id'),
                        service=item.get('service'),
                        environment=item.get('environment'),
                        cluster=item.get('cluster'),
                        namespace=item.get('namespace'),
                        pod=item.get('pod'),
                        container=item.get('container'),
                        file_path=item.get('file_path'),
                        commit_sha=item.get('commit_sha'),
                        tool=item.get('tool'),
                        configs=item.get('configs'),
                        docs_faq=item.get('docs_faq'),
                        status=item.get('status'),
                        resolution=item.get('resolution'),
                        severity=item.get('severity'),
                        category=item.get('category'),
                        priority=item.get('priority'),
                        tags=item.get('tags'),
                        created_at=created_at,
                        updated_at=updated_at
                    )
                    sample_logs.append(log_data)
                
                print(f"✓ Loaded {len(sample_logs)} incident records from JSON file")
                
            except Exception as e:
                print(f"✗ Failed to load JSON data: {e}")
                print("Using fallback sample data...")
                # Fallback to minimal sample data
                sample_logs = [
                    LogData(
                        log_id="LOG-001",
                        title="Login Issues - Authentication",
                        description="Users cannot login; authentication error 401 returned",
                        source_type="log",
                        service="auth-service",
                        environment="prod",
                        cluster="gke-prod",
                        namespace="auth",
                        pod="auth-7f9c8",
                        container="auth",
                        severity="error",
                        resolution="Reset user session token, invalidate caches",
                        incident_id="INC-001",
                        configs="AUTH_CLIENT_SECRET not set",
                        docs_faq="Refer to auth troubleshooting guide",
                        status="Resolved",
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        category="Authentication",
                        priority="High",
                        tags=["oauth", "401"]
                    )
                ]
            
            # Add logs to vector store
            print("\n📝 Adding sample logs to vector store...")
            agent.add_logs_to_vector_store(sample_logs)
                                                
            # Perform vector searches
            print("\n🔍 Performing vector searches...")
            
            # Single query search
            query1 = "login authentication problems"
            results1 = agent.vector_search(query1, k=3)
            
            print(f"\n📋 Search Results for: '{query1}'")
            print("=" * 50)
            for i, result in enumerate(results1, 1):
                print(f"\n{i}. Log ID: {result.log_id}")
                print(f"   Content: {result.content}")
                print(f"   Similarity Score: {result.similarity_score:.4f}")
                print(f"   Priority: {result.metadata.get('priority', 'N/A')}")
                print(f"   Status: {result.metadata.get('status', 'N/A')}")
            
            # Batch search
            queries = [
                "performance issues",
                "password reset problems", 
                "database connection errors"
            ]
            
            print(f"\n📋 Batch Search Results")
            print("=" * 50)
            batch_results = agent.batch_vector_search(queries, k=2)
            
            for query, results in batch_results.items():
                print(f"\n🔍 Query: '{query}'")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result.log_id} (Score: {result.similarity_score:.4f})")
            
            # Analyze results
            print(f"\n📊 Analysis of Search Results")
            print("=" * 50)
            analysis = agent.analyze_search_results(results1)
            for key, value in analysis.items():
                print(f"{key}: {value}")
            
            print("\n✅ Vector Search Agent demonstration completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error in main execution: {e}")
            raise


if __name__ == "__main__":
    main()
