# embedding_agent.py
import os
import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any, Literal
import numpy as np
from dotenv import load_dotenv

# Google Cloud imports
from google.cloud import bigquery
from google.oauth2 import service_account

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
# --------- Dataclass ---------
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


# --------- Embedding generator ---------
# --------- Embedding generator ---------
class TextEmbeddingGenerator:
    def __init__(self, model_name: str = "text-embedding-004", api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_GEMINI_API_KEY not found in env")
        # LangChain GoogleGenerativeAI wrapper used as embeddings provider
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=self.api_key)

        # Text splitter helps chunk long logs/configs/docs
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

        # 🔹 System prompt for embedding consistency
        self.SYSTEM_PROMPT = (
            "You are embedding DevOps / SaaS incident data for similarity search. "
            "Focus on root cause analysis, remediation steps, configs, and operational context. "
            "Capture semantic meaning of errors, resolutions, and logs rather than surface words."
        )

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text string using LangChain wrapper."""
        try:
            # prepend system prompt to ensure embeddings are aligned with RCA task
            enriched_text = f"{self.SYSTEM_PROMPT}\n\n{text}"
            return self.embedding_model.embed_query(enriched_text)
        except Exception as e:
            print(f"[Embedding] error for text (truncated): {str(e)}")
            return None

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for t in texts:
            emb = self.generate_embedding(t)
            if emb:
                embeddings.append(emb)
            else:
                embeddings.append([0.0] * 1536)  # safe fallback vector
        return embeddings


    def prepare_text_for_embedding(self, log: LogData) -> str:
        # Structured content to improve embedding quality for RCA
        parts = [
            f"Title: {log.title}",
            f"SourceType: {log.source_type}",
            f"Description: {log.description}",
        ]
        if log.configs:
            parts.append(f"Configs:\n{log.configs}")
        if log.docs_faq:
            parts.append(f"DocsOrFAQ:\n{log.docs_faq}")
        if log.resolution:
            parts.append(f"Resolution:\n{log.resolution}")
        # Operational context
        context_bits = []
        for name, val in [
            ("IncidentID", log.incident_id),
            ("Service", log.service),
            ("Environment", log.environment),
            ("Cluster", log.cluster),
            ("Namespace", log.namespace),
            ("Pod", log.pod),
            ("Container", log.container),
            ("FilePath", log.file_path),
            ("CommitSHA", log.commit_sha),
            ("Tool", log.tool),
            ("Status", log.status),
            ("Severity", log.severity),
            ("Category", log.category),
            ("Priority", log.priority),
        ]:
            if val:
                context_bits.append(f"{name}: {val}")
        if log.tags:
            context_bits.append(f"Tags: {', '.join(log.tags)}")
        if context_bits:
            parts.append("Context:\n" + "\n".join(context_bits))
        return "\n\n".join(parts)


# --------- EmbeddingAgent / BigQuery wrapper ---------
class EmbeddingAgent:
    def __init__(self, model_name: str = "text-embedding-004", project_id: str = None, dataset_id: str = "log_data", table_id: str = "logs_table"):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.embedding_generator = TextEmbeddingGenerator(model_name=model_name)
        
        # Initialize BigQuery client
        self.bq_client = self._initialize_bigquery_client()
        
        # Keep a list of all Document objects for stats
        self.all_documents: List[Document] = []

    def _initialize_bigquery_client(self) -> bigquery.Client:
        """Initialize BigQuery client with authentication."""
        try:
            # Try to use service account key if provided
            service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if service_account_path and os.path.exists(service_account_path):
                credentials = service_account.Credentials.from_service_account_file(service_account_path)
                client = bigquery.Client(credentials=credentials, project=self.project_id)
            else:
                # Use default credentials (e.g., from gcloud auth)
                client = bigquery.Client(project=self.project_id)
            
            print(f"[BigQuery] Connected to project: {self.project_id}")
            return client
        except Exception as e:
            print(f"[BigQuery] Failed to initialize client: {e}")
            raise

    def _make_document(self, log: LogData) -> Document:
        text = self.embedding_generator.prepare_text_for_embedding(log)
        meta = {
            "log_id": log.log_id,
            "title": log.title,
            "description": log.description,
            "source_type": log.source_type,
            "incident_id": log.incident_id,
            "service": log.service,
            "environment": log.environment,
            "cluster": log.cluster,
            "namespace": log.namespace,
            "pod": log.pod,
            "container": log.container,
            "file_path": log.file_path,
            "commit_sha": log.commit_sha,
            "tool": log.tool,
            "has_configs": bool(log.configs),
            "has_docs_faq": bool(log.docs_faq),
            "status": log.status,
            "resolution": log.resolution,
            "has_resolution": bool(log.resolution),
            "severity": log.severity,
            "category": log.category,
            "priority": log.priority,
            "tags": log.tags or [],
            "created_at": log.created_at.isoformat() if log.created_at else None,
            "updated_at": log.updated_at.isoformat() if log.updated_at else None,
        }
        return Document(page_content=text, metadata=meta)

    def process_log(self, log: LogData) -> Dict[str, Any]:
        text = self.embedding_generator.prepare_text_for_embedding(log)
        emb = self.embedding_generator.generate_embedding(text)
        if emb is None:
            raise RuntimeError(f"Failed to generate embedding for {log.log_id}")
        doc = self._make_document(log)
        return {"log_id": log.log_id, "embedding": emb, "document": doc, "text": text}

    def process_logs_batch(self, logs: List[LogData]) -> List[Dict[str, Any]]:
        results = []
        for log in logs:
            try:
                results.append(self.process_log(log))
            except Exception as e:
                print(f"[process_logs_batch] skipping {log.log_id}: {e}")
        return results


    def insert_logs_to_bigquery(self, processed_logs: List[Dict[str, Any]]) -> None:
        """Insert processed logs with embeddings into BigQuery."""
        if not processed_logs:
            print("[insert_logs_to_bigquery] no logs to insert")
            return

        table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
        
        # Prepare rows for BigQuery
        rows_to_insert = []
        for log_data in processed_logs:
            doc = log_data["document"]
            embedding = log_data["embedding"]
            
            row = {
                "log_id": doc.metadata.get("log_id"),
                "title": doc.metadata.get("title"),
                "description": doc.metadata.get("description"),
                "source_type": doc.metadata.get("source_type"),
                "incident_id": doc.metadata.get("incident_id"),
                "service": doc.metadata.get("service"),
                "environment": doc.metadata.get("environment"),
                "cluster": doc.metadata.get("cluster"),
                "namespace": doc.metadata.get("namespace"),
                "pod": doc.metadata.get("pod"),
                "container": doc.metadata.get("container"),
                "file_path": doc.metadata.get("file_path"),
                "commit_sha": doc.metadata.get("commit_sha"),
                "tool": doc.metadata.get("tool"),
                "configs": doc.metadata.get("configs"),
                "docs_faq": doc.metadata.get("docs_faq"),
                "status": doc.metadata.get("status"),
                "resolution": doc.metadata.get("resolution"),
                "severity": doc.metadata.get("severity"),
                "category": doc.metadata.get("category"),
                "priority": doc.metadata.get("priority"),
                "tags": doc.metadata.get("tags", []),
                "created_at": doc.metadata.get("created_at"),
                "updated_at": doc.metadata.get("updated_at"),
                "embedding": embedding,
                "content_for_embedding": doc.page_content,
            }
            rows_to_insert.append(row)

        # Insert rows
        try:
            errors = self.bq_client.insert_rows_json(table_id, rows_to_insert)
            if errors:
                print(f"[BigQuery] Errors inserting rows: {errors}")
            else:
                print(f"[BigQuery] Successfully inserted {len(rows_to_insert)} rows")
        except Exception as e:
            print(f"[BigQuery] Error inserting rows: {e}")
            raise

    def search_similar_logs(self, query: str, k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar logs using BigQuery vector similarity search."""
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.generate_embedding(query)
            if query_embedding is None:
                print("[search_similar_logs] Failed to generate query embedding")
                return []

            # Build the BigQuery SQL query
            table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            
            # Base query with vector similarity search
            base_query = f"""
            SELECT 
                log_id,
                title,
                description,
                source_type,
                incident_id,
                service,
                environment,
                cluster,
                namespace,
                pod,
                container,
                file_path,
                commit_sha,
                tool,
                configs,
                docs_faq,
                status,
                resolution,
                severity,
                category,
                priority,
                tags,
                created_at,
                updated_at,
                content_for_embedding,
                ML.DISTANCE(embedding, @query_embedding, 'COSINE') as similarity_score
            FROM `{table_id}`
            """
            
            # Add WHERE clause for metadata filtering
            where_conditions = []
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
            formatted = []
            for row in results:
                formatted.append({
                    "log_id": row.log_id,
                    "title": row.title,
                    "source_type": row.source_type,
                    "content": row.content_for_embedding,
                    "similarity_score": float(row.similarity_score),
                    "metadata": {
                        "log_id": row.log_id,
                        "title": row.title,
                        "description": row.description,
                        "source_type": row.source_type,
                        "incident_id": row.incident_id,
                        "service": row.service,
                        "environment": row.environment,
                        "cluster": row.cluster,
                        "namespace": row.namespace,
                        "pod": row.pod,
                        "container": row.container,
                        "file_path": row.file_path,
                        "commit_sha": row.commit_sha,
                        "tool": row.tool,
                        "configs": row.configs,
                        "docs_faq": row.docs_faq,
                        "status": row.status,
                        "resolution": row.resolution,
                        "severity": row.severity,
                        "category": row.category,
                        "priority": row.priority,
                        "tags": list(row.tags) if row.tags else [],
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    }
                })
            
            return formatted
            
        except Exception as e:
            print(f"[search_similar_logs] error: {e}")
            return []

    def find_similar_resolutions(self, log_description: str, k: int = 3) -> List[Dict[str, Any]]:
        """Find similar logs that contain resolutions or are incident docs."""
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.generate_embedding(log_description)
            if query_embedding is None:
                print("[find_similar_resolutions] Failed to generate query embedding")
                return []

            # Build the BigQuery SQL query with resolution filtering
            table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            
            query = f"""
            SELECT 
                log_id,
                title,
                description,
                source_type,
                incident_id,
                service,
                environment,
                cluster,
                namespace,
                pod,
                container,
                file_path,
                commit_sha,
                tool,
                configs,
                docs_faq,
                status,
                resolution,
                severity,
                category,
                priority,
                tags,
                created_at,
                updated_at,
                content_for_embedding,
                ML.DISTANCE(embedding, @query_embedding, 'COSINE') as similarity_score
            FROM `{table_id}`
            WHERE (resolution IS NOT NULL AND resolution != '') 
               OR source_type IN ('doc', 'incident')
            ORDER BY similarity_score ASC
            LIMIT {k * 5}
            """
            
            # Execute query
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("query_embedding", "ARRAY<FLOAT64>", query_embedding)
                ]
            )
            
            query_job = self.bq_client.query(query, job_config=job_config)
            results = query_job.result()
            
            # Format results
            formatted = []
            for row in results:
                formatted.append({
                    "log_id": row.log_id,
                    "title": row.title,
                    "source_type": row.source_type,
                    "content": row.content_for_embedding,
                    "similarity_score": float(row.similarity_score),
                    "metadata": {
                        "log_id": row.log_id,
                        "title": row.title,
                        "description": row.description,
                        "source_type": row.source_type,
                        "incident_id": row.incident_id,
                        "service": row.service,
                        "environment": row.environment,
                        "cluster": row.cluster,
                        "namespace": row.namespace,
                        "pod": row.pod,
                        "container": row.container,
                        "file_path": row.file_path,
                        "commit_sha": row.commit_sha,
                        "tool": row.tool,
                        "configs": row.configs,
                        "docs_faq": row.docs_faq,
                        "status": row.status,
                        "resolution": row.resolution,
                        "severity": row.severity,
                        "category": row.category,
                        "priority": row.priority,
                        "tags": list(row.tags) if row.tags else [],
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    }
                })
            
            return formatted[:k]  # Return only the requested number of results
            
        except Exception as e:
            print(f"[find_similar_resolutions] error: {e}")
            return []

    def get_log_statistics(self) -> Dict[str, Any]:
        """Get statistics from BigQuery table."""
        try:
            table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            
            # Query for basic statistics
            stats_query = f"""
            SELECT 
                COUNT(*) as total_artifacts,
                COUNTIF(resolution IS NOT NULL AND resolution != '') as resolved_items,
                COUNTIF(source_type = 'log') as log_count,
                COUNTIF(source_type = 'config') as config_count,
                COUNTIF(source_type = 'doc') as doc_count,
                COUNTIF(source_type = 'incident') as incident_count,
                COUNTIF(severity = 'critical') as critical_count,
                COUNTIF(severity = 'error') as error_count,
                COUNTIF(severity = 'warn') as warn_count,
                COUNTIF(severity = 'info') as info_count
            FROM `{table_id}`
            """
            
            query_job = self.bq_client.query(stats_query)
            results = list(query_job.result())
            
            if not results:
                return {"error": "No data found in BigQuery table"}
            
            row = results[0]
            
            # Query for category breakdown
            category_query = f"""
            SELECT category, COUNT(*) as count
            FROM `{table_id}`
            WHERE category IS NOT NULL
            GROUP BY category
            ORDER BY count DESC
            """
            
            category_job = self.bq_client.query(category_query)
            category_results = list(category_job.result())
            
            by_category = {row.category: row.count for row in category_results}
            
            return {
                "total_artifacts": row.total_artifacts,
                "resolved_items": row.resolved_items,
                "unresolved_items": row.total_artifacts - row.resolved_items,
                "by_source_type": {
                    "log": row.log_count,
                    "config": row.config_count,
                    "doc": row.doc_count,
                    "incident": row.incident_count
                },
                "by_severity": {
                    "critical": row.critical_count,
                    "error": row.error_count,
                    "warn": row.warn_count,
                    "info": row.info_count
                },
                "by_category": by_category,
                "bigquery_available": True
            }
            
        except Exception as e:
            print(f"[get_log_statistics] error: {e}")
            return {
                "error": str(e),
                "bigquery_available": False
            }

def load_logs_from_bigquery(agent: EmbeddingAgent, limit: Optional[int] = None) -> List[LogData]:
    """Load logs from BigQuery table into LogData objects."""
    try:
        table_id = f"{agent.project_id}.{agent.dataset_id}.{agent.table_id}"
        
        query = f"""
        SELECT 
            log_id,
            title,
            description,
            source_type,
            incident_id,
            service,
            environment,
            cluster,
            namespace,
            pod,
            container,
            file_path,
            commit_sha,
            tool,
            configs,
            docs_faq,
            status,
            resolution,
            severity,
            category,
            priority,
            tags,
            created_at,
            updated_at
        FROM `{table_id}`
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        query_job = agent.bq_client.query(query)
        results = query_job.result()
        
        logs: List[LogData] = []
        for row in results:
            try:
                logs.append(
                    LogData(
                        log_id=row.log_id,
                        title=row.title,
                        description=row.description,
                        source_type=row.source_type,
                        incident_id=row.incident_id,
                        service=row.service,
                        environment=row.environment,
                        cluster=row.cluster,
                        namespace=row.namespace,
                        pod=row.pod,
                        container=row.container,
                        file_path=row.file_path,
                        commit_sha=row.commit_sha,
                        tool=row.tool,
                        configs=row.configs,
                        docs_faq=row.docs_faq,
                        status=row.status,
                        resolution=row.resolution,
                        severity=row.severity,
                        category=row.category,
                        priority=row.priority,
                        tags=list(row.tags) if row.tags else None,
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                    )
                )
            except Exception as e:
                print(f"[load_logs_from_bigquery] skipping invalid entry: {e}")
        
        print(f"[load_logs_from_bigquery] loaded {len(logs)} logs from BigQuery")
        return logs
        
    except Exception as e:
        print(f"[load_logs_from_bigquery] error: {e}")
        return []

def load_logs_from_json(file_path: str) -> List[LogData]:
    """Load logs from your JSON dataset into LogData objects."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    logs: List[LogData] = []
    for item in raw_data:
        try:
            logs.append(
                LogData(
                    log_id=item.get("log_id", ""),
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    source_type=item.get("source_type", "log"),
                    incident_id=item.get("incident_id"),
                    service=item.get("service"),
                    environment=item.get("environment"),
                    cluster=item.get("cluster"),
                    namespace=item.get("namespace"),
                    pod=item.get("pod"),
                    container=item.get("container"),
                    file_path=item.get("file_path"),
                    commit_sha=item.get("commit_sha"),
                    tool=item.get("tool"),
                    configs=item.get("configs"),
                    docs_faq=item.get("docs_faq"),
                    status=item.get("status"),
                    resolution=item.get("resolution"),
                    severity=item.get("severity"),
                    category=item.get("category"),
                    priority=item.get("priority"),
                    tags=item.get("tags"),
                    # parse datetime if available
                    created_at=datetime.fromisoformat(item["created_at"]) if item.get("created_at") else None,
                    updated_at=datetime.fromisoformat(item["updated_at"]) if item.get("updated_at") else None,
                )
            )
        except Exception as e:
            print(f"[load_logs_from_json] skipping invalid entry: {e}")
    return logs


def main():
    """Main function to demonstrate BigQuery embedding agent functionality."""
    # Initialize the agent with BigQuery configuration
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "big-station-472112-i1")
    agent = EmbeddingAgent(
        project_id=project_id,
        dataset_id="log_data",
        table_id="logs_table"
    )


    # Option 1: Load from JSON file and insert into BigQuery (for initial setup)
    dataset_path = r"G:\OneDrive\Desktop\google cloud AI hackathon\incident_dataset.json"
    if os.path.exists(dataset_path):
        print("Loading data from JSON file...")
        logs = load_logs_from_json(dataset_path)
        
        # Process logs and insert into BigQuery
        processed = agent.process_logs_batch(logs)
        print(f"Processed {len(processed)} artifacts")
        
        # Insert into BigQuery
        agent.insert_logs_to_bigquery(processed)
    else:
        print("JSON file not found, loading from BigQuery...")
        # Option 2: Load existing data from BigQuery
        logs = load_logs_from_bigquery(agent, limit=100)  # Load first 100 records for testing

    # Test search functionality
    print("\n" + "="*50)
    print("TESTING SEARCH FUNCTIONALITY")
    print("="*50)
    
    query = "authentication error 401 users cannot login auth service"
    similar = agent.search_similar_logs(query, k=3, filter_metadata={"environment": "prod"})
    print("Similar results (prod only):")
    for r in similar:
        print(f"- {r['log_id']} | {r['source_type']} | {r['title']} | score: {r['similarity_score']:.4f}")

    print("\nSimilar items with resolutions or incident/docs (for RCA):")
    sr = agent.find_similar_resolutions("Authentication failures and login issues", k=3)
    for r in sr:
        print("-", r["log_id"], r["source_type"], r["title"]) 

    print("\nStatistics:")
    stats = agent.get_log_statistics()
    print(json.dumps(stats, indent=2))
    
    return agent  # Return the agent for use in other files


def create_agent(project_id: str = None, dataset_id: str = "log_data", table_id: str = "logs_table") -> EmbeddingAgent:
    """Create and initialize an EmbeddingAgent instance.
    
    Args:
        project_id: GCP project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
        
    Returns:
        Initialized EmbeddingAgent instance
    """
    if project_id is None:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "big-station-472112-i1")
    
    agent = EmbeddingAgent(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id
    )
    
    return agent


def ensure_vector_store_ready(agent: EmbeddingAgent) -> bool:
    """Ensure the vector store is ready for the workflow.
    
    Args:
        agent: EmbeddingAgent instance
        
    Returns:
        True if vector store is ready, False otherwise
    """
    try:
        # Check if BigQuery table exists and has data
        stats = agent.get_log_statistics()
        
        if "error" in stats:
            print(f"[ensure_vector_store_ready] BigQuery error: {stats['error']}")
            return False
        
        if stats.get("total_artifacts", 0) == 0:
            print("[ensure_vector_store_ready] No data in BigQuery table")
            return False
        
        print(f"[ensure_vector_store_ready] Vector store ready with {stats['total_artifacts']} artifacts")
        return True
        
    except Exception as e:
        print(f"[ensure_vector_store_ready] Error: {e}")
        return False


def prepare_embeddings_for_workflow(agent: EmbeddingAgent, 
                                  data_source: str = "bigquery",
                                  limit: int = None) -> List[Dict[str, Any]]:
    """Prepare embeddings for the workflow by ensuring data is available.
    
    Args:
        agent: EmbeddingAgent instance
        data_source: Source of data ("bigquery" or "json")
        limit: Limit number of records to process
        
    Returns:
        List of processed log data with embeddings
    """
    try:
        if data_source == "bigquery":
            # Load from BigQuery
            logs = load_logs_from_bigquery(agent, limit=limit)
        else:
            # Load from JSON file
            dataset_path = "incident_dataset.json"
            if not os.path.exists(dataset_path):
                print(f"[prepare_embeddings_for_workflow] JSON file not found: {dataset_path}")
                return []
            logs = load_logs_from_json(dataset_path)
            if limit:
                logs = logs[:limit]
        
        if not logs:
            print("[prepare_embeddings_for_workflow] No logs to process")
            return []
        
        # Process logs and generate embeddings
        processed_logs = agent.process_logs_batch(logs)
        
        # Insert into BigQuery if not already there
        if data_source == "json":
            agent.insert_logs_to_bigquery(processed_logs)
        
        print(f"[prepare_embeddings_for_workflow] Processed {len(processed_logs)} logs")
        return processed_logs
        
    except Exception as e:
        print(f"[prepare_embeddings_for_workflow] Error: {e}")
        return []


def search_incidents(agent: EmbeddingAgent, query: str, k: int = 5, **filters) -> List[Dict[str, Any]]:
    """Search for similar incidents using the embedding agent.
    
    Args:
        agent: EmbeddingAgent instance
        query: Search query string
        k: Number of results to return
        **filters: Additional metadata filters (e.g., environment="prod")
        
    Returns:
        List of similar incident results
    """
    return agent.search_similar_logs(query, k=k, filter_metadata=filters if filters else None)


def find_resolutions(agent: EmbeddingAgent, description: str, k: int = 3) -> List[Dict[str, Any]]:
    """Find similar incidents with resolutions.
    
    Args:
        agent: EmbeddingAgent instance
        description: Incident description to search for
        k: Number of results to return
        
    Returns:
        List of similar incidents with resolutions
    """
    return agent.find_similar_resolutions(description, k=k)


def get_incident_stats(agent: EmbeddingAgent) -> Dict[str, Any]:
    """Get incident statistics from BigQuery.
    
    Args:
        agent: EmbeddingAgent instance
        
    Returns:
        Dictionary containing incident statistics
    """
    return agent.get_log_statistics()


if __name__ == "__main__":
    main()