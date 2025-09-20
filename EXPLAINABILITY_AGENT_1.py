# explainability_agent.py
import os
import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from collections import Counter
from dotenv import load_dotenv

# Google Cloud imports
from google.cloud import bigquery
from google.oauth2 import service_account

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

load_dotenv()

# --------- Dataclass for Explainability Results ---------
@dataclass
class EvidenceTicket:
    """Represents an evidence ticket with similarity score and metadata."""
    ticket_id: str
    title: str
    similarity_score: float
    confidence_percentage: int
    source_type: str
    resolution: Optional[str] = None
    category: Optional[str] = None
    severity: Optional[str] = None
    environment: Optional[str] = None
    service: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class ExplainabilityResult:
    """Complete explainability result with confidence scores and evidence."""
    query: str
    overall_confidence: int
    evidence_tickets: List[EvidenceTicket]
    common_fixes: List[Dict[str, Any]]
    reasoning: str
    recommendations: List[str]
    transparency_score: float
    metadata: Dict[str, Any]

# --------- Explainability Agent ---------
class ExplainabilityAgent:
    """
    Explainability Agent that provides confidence scores and evidence tickets
    for incident analysis with transparency and trust features.
    """
    
    def __init__(self, project_id: str = None, dataset_id: str = "incident_data", table_id: str = "incidents"):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.embedding_generator = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004", 
            google_api_key=os.getenv("GOOGLE_GEMINI_API_KEY")
        )
        
        # Initialize BigQuery client
        self.bq_client = self._initialize_bigquery_client()
        
        # Confidence thresholds
        self.CONFIDENCE_THRESHOLDS = {
            "high": 0.8,      # 80%+ similarity
            "medium": 0.6,    # 60-80% similarity
            "low": 0.4,       # 40-60% similarity
            "very_low": 0.0   # <40% similarity
        }
        
        # System prompt for explainability
        self.EXPLAINABILITY_PROMPT = (
            "You are an explainability agent for DevOps incident analysis. "
            "Provide clear, transparent explanations of why certain tickets are similar, "
            "what common patterns exist, and what fixes are typically applied. "
            "Focus on building trust through detailed reasoning and evidence."
        )

    def _initialize_bigquery_client(self) -> bigquery.Client:
        """Initialize BigQuery client with authentication."""
        try:
            service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if service_account_path and os.path.exists(service_account_path):
                credentials = service_account.Credentials.from_service_account_file(service_account_path)
                client = bigquery.Client(credentials=credentials, project=self.project_id)
            else:
                client = bigquery.Client(project=self.project_id)
            
            print(f"[ExplainabilityAgent] Connected to BigQuery project: {self.project_id}")
            return client
        except Exception as e:
            print(f"[ExplainabilityAgent] Failed to initialize BigQuery client: {e}")
            raise

    def _calculate_confidence_percentage(self, similarity_score: float) -> int:
        """Convert similarity score to confidence percentage."""
        # Convert cosine distance to similarity (1 - distance)
        similarity = 1 - similarity_score
        # Scale to percentage and ensure it's within 0-100
        confidence = max(0, min(100, int(similarity * 100)))
        return confidence

    def _get_confidence_level(self, confidence_percentage: int) -> str:
        """Get confidence level based on percentage."""
        if confidence_percentage >= 80:
            return "high"
        elif confidence_percentage >= 60:
            return "medium"
        elif confidence_percentage >= 40:
            return "low"
        else:
            return "very_low"

    def _search_similar_incidents(self, query: str, k: int = 10, 
                                filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar incidents using BigQuery vector similarity search."""
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.embed_query(query)
            if query_embedding is None:
                print("[_search_similar_incidents] Failed to generate query embedding")
                return []

            # Build the BigQuery SQL query
            table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            
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
                    "content_for_embedding": row.content_for_embedding,
                    "similarity_score": float(row.similarity_score)
                })
            
            return formatted
            
        except Exception as e:
            print(f"[_search_similar_incidents] error: {e}")
            return []

    def _create_evidence_tickets(self, similar_incidents: List[Dict[str, Any]]) -> List[EvidenceTicket]:
        """Create evidence tickets from similar incidents."""
        evidence_tickets = []
        
        for incident in similar_incidents:
            confidence_percentage = self._calculate_confidence_percentage(incident["similarity_score"])
            confidence_level = self._get_confidence_level(confidence_percentage)
            
            evidence_ticket = EvidenceTicket(
                ticket_id=incident["log_id"],
                title=incident["title"],
                similarity_score=incident["similarity_score"],
                confidence_percentage=confidence_percentage,
                source_type=incident["source_type"],
                resolution=incident.get("resolution"),
                category=incident.get("category"),
                severity=incident.get("severity"),
                environment=incident.get("environment"),
                service=incident.get("service"),
                created_at=incident.get("created_at"),
                metadata={
                    "description": incident.get("description"),
                    "incident_id": incident.get("incident_id"),
                    "status": incident.get("status"),
                    "priority": incident.get("priority"),
                    "tags": incident.get("tags", []),
                    "cluster": incident.get("cluster"),
                    "namespace": incident.get("namespace"),
                    "pod": incident.get("pod"),
                    "container": incident.get("container"),
                    "confidence_level": confidence_level
                }
            )
            evidence_tickets.append(evidence_ticket)
        
        # Sort by confidence percentage (highest first)
        evidence_tickets.sort(key=lambda x: x.confidence_percentage, reverse=True)
        return evidence_tickets

    def _analyze_common_fixes(self, evidence_tickets: List[EvidenceTicket]) -> List[Dict[str, Any]]:
        """Analyze common fixes from evidence tickets."""
        fixes = []
        resolutions = [ticket.resolution for ticket in evidence_tickets if ticket.resolution]
        
        if not resolutions:
            return fixes
        
        # Count resolution patterns
        resolution_counter = Counter(resolutions)
        
        # Extract common patterns
        for resolution, count in resolution_counter.most_common(5):
            if resolution and len(resolution.strip()) > 10:  # Filter out very short resolutions
                fixes.append({
                    "fix_description": resolution,
                    "frequency": count,
                    "percentage": round((count / len(resolutions)) * 100, 1),
                    "tickets_using": [ticket.ticket_id for ticket in evidence_tickets if ticket.resolution == resolution]
                })
        
        return fixes

    def _generate_reasoning(self, query: str, evidence_tickets: List[EvidenceTicket], 
                          common_fixes: List[Dict[str, Any]]) -> str:
        """Generate human-readable reasoning for the explainability result."""
        if not evidence_tickets:
            return "No similar incidents found in the database."
        
        # Get top evidence tickets
        top_tickets = evidence_tickets[:3]
        high_confidence_tickets = [t for t in evidence_tickets if t.confidence_percentage >= 80]
        medium_confidence_tickets = [t for t in evidence_tickets if 60 <= t.confidence_percentage < 80]
        
        reasoning_parts = []
        
        # Overall similarity analysis
        reasoning_parts.append(f"Found {len(evidence_tickets)} similar incidents to your query: '{query}'")
        
        if high_confidence_tickets:
            reasoning_parts.append(f"• {len(high_confidence_tickets)} high-confidence matches (80%+ similarity)")
        if medium_confidence_tickets:
            reasoning_parts.append(f"• {len(medium_confidence_tickets)} medium-confidence matches (60-80% similarity)")
        
        # Top similar tickets
        reasoning_parts.append("\nMost similar tickets:")
        for i, ticket in enumerate(top_tickets, 1):
            reasoning_parts.append(f"  {i}. Ticket #{ticket.ticket_id[:8]}... - {ticket.confidence_percentage}% similar")
            reasoning_parts.append(f"     Title: {ticket.title}")
            if ticket.category:
                reasoning_parts.append(f"     Category: {ticket.category}")
            if ticket.environment:
                reasoning_parts.append(f"     Environment: {ticket.environment}")
        
        # Common fixes analysis
        if common_fixes:
            reasoning_parts.append(f"\nCommon fixes identified:")
            for i, fix in enumerate(common_fixes[:3], 1):
                reasoning_parts.append(f"  {i}. {fix['fix_description'][:100]}{'...' if len(fix['fix_description']) > 100 else ''}")
                reasoning_parts.append(f"     Used in {fix['frequency']} tickets ({fix['percentage']}%)")
        
        return "\n".join(reasoning_parts)

    def _generate_recommendations(self, evidence_tickets: List[EvidenceTicket], 
                                common_fixes: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on evidence."""
        recommendations = []
        
        if not evidence_tickets:
            recommendations.append("No similar incidents found. Consider creating a new incident category.")
            return recommendations
        
        # High confidence recommendations
        high_confidence_tickets = [t for t in evidence_tickets if t.confidence_percentage >= 80]
        if high_confidence_tickets:
            recommendations.append(f"High confidence match found! Review Ticket #{high_confidence_tickets[0].ticket_id[:8]}... for immediate resolution.")
        
        # Category-based recommendations
        categories = [t.category for t in evidence_tickets if t.category]
        if categories:
            most_common_category = Counter(categories).most_common(1)[0][0]
            recommendations.append(f"Most incidents are in the '{most_common_category}' category. Check related documentation.")
        
        # Environment-specific recommendations
        environments = [t.environment for t in evidence_tickets if t.environment]
        if environments:
            most_common_env = Counter(environments).most_common(1)[0][0]
            recommendations.append(f"Most similar incidents occurred in '{most_common_env}' environment. Check environment-specific configurations.")
        
        # Resolution-based recommendations
        if common_fixes:
            top_fix = common_fixes[0]
            recommendations.append(f"Most common fix: '{top_fix['fix_description'][:100]}{'...' if len(top_fix['fix_description']) > 100 else ''}'")
            recommendations.append(f"This fix was successful in {top_fix['frequency']} similar cases.")
        
        # Severity-based recommendations
        critical_tickets = [t for t in evidence_tickets if t.severity == 'critical']
        if critical_tickets:
            recommendations.append(f"⚠️  {len(critical_tickets)} critical incidents found. Prioritize immediate action.")
        
        return recommendations

    def _calculate_transparency_score(self, evidence_tickets: List[EvidenceTicket], 
                                    common_fixes: List[Dict[str, Any]]) -> float:
        """Calculate transparency score based on available evidence and explanations."""
        score = 0.0
        max_score = 100.0
        
        # Evidence availability (40 points)
        if evidence_tickets:
            score += 20  # Base points for having evidence
            high_confidence_count = len([t for t in evidence_tickets if t.confidence_percentage >= 80])
            score += min(20, high_confidence_count * 5)  # Up to 20 points for high confidence evidence
        
        # Resolution availability (30 points)
        resolved_tickets = [t for t in evidence_tickets if t.resolution]
        if resolved_tickets:
            resolution_rate = len(resolved_tickets) / len(evidence_tickets)
            score += resolution_rate * 30
        
        # Common fixes availability (20 points)
        if common_fixes:
            score += min(20, len(common_fixes) * 4)  # Up to 20 points for common fixes
        
        # Metadata completeness (10 points)
        complete_metadata_count = 0
        for ticket in evidence_tickets:
            metadata_completeness = sum(1 for v in [
                ticket.category, ticket.severity, ticket.environment, ticket.service
            ] if v is not None)
            complete_metadata_count += metadata_completeness / 4
        
        if evidence_tickets:
            avg_metadata_completeness = complete_metadata_count / len(evidence_tickets)
            score += avg_metadata_completeness * 10
        
        return min(score, max_score)

    def explain_incident(self, query: str, k: int = 10, 
                        filter_metadata: Optional[Dict] = None) -> ExplainabilityResult:
        """
        Main method to explain an incident with confidence scores and evidence tickets.
        
        Args:
            query: Incident description or query
            k: Number of similar incidents to retrieve
            filter_metadata: Optional metadata filters
            
        Returns:
            ExplainabilityResult with confidence scores and evidence
        """
        print(f"[ExplainabilityAgent] Analyzing incident: '{query}'")
        
        # Search for similar incidents
        similar_incidents = self._search_similar_incidents(query, k, filter_metadata)
        
        # Create evidence tickets
        evidence_tickets = self._create_evidence_tickets(similar_incidents)
        
        # Analyze common fixes
        common_fixes = self._analyze_common_fixes(evidence_tickets)
        
        # Calculate overall confidence
        if evidence_tickets:
            overall_confidence = int(np.mean([t.confidence_percentage for t in evidence_tickets[:5]]))
        else:
            overall_confidence = 0
        
        # Generate reasoning and recommendations
        reasoning = self._generate_reasoning(query, evidence_tickets, common_fixes)
        recommendations = self._generate_recommendations(evidence_tickets, common_fixes)
        
        # Calculate transparency score
        transparency_score = self._calculate_transparency_score(evidence_tickets, common_fixes)
        
        # Create result
        result = ExplainabilityResult(
            query=query,
            overall_confidence=overall_confidence,
            evidence_tickets=evidence_tickets,
            common_fixes=common_fixes,
            reasoning=reasoning,
            recommendations=recommendations,
            transparency_score=transparency_score,
            metadata={
                "total_evidence_tickets": len(evidence_tickets),
                "high_confidence_tickets": len([t for t in evidence_tickets if t.confidence_percentage >= 80]),
                "medium_confidence_tickets": len([t for t in evidence_tickets if 60 <= t.confidence_percentage < 80]),
                "low_confidence_tickets": len([t for t in evidence_tickets if t.confidence_percentage < 60]),
                "resolved_tickets": len([t for t in evidence_tickets if t.resolution]),
                "analysis_timestamp": datetime.now().isoformat()
            }
        )
        
        print(f"[ExplainabilityAgent] Analysis complete. Overall confidence: {overall_confidence}%")
        return result
    
    def process_from_resolution_agent(self, 
                                    resolution_data: Dict[str, Any],
                                    query: str = None) -> ExplainabilityResult:
        """
        Process data from resolution agent and provide explainability analysis.
        This method takes the output from the resolution agent and enhances it with explainability.
        
        Args:
            resolution_data: Data from resolution agent (from process_for_explainability_agent)
            query: Original query (if not in resolution_data)
            
        Returns:
            ExplainabilityResult with enhanced explainability
        """
        try:
            if "error" in resolution_data:
                print(f"[process_from_resolution_agent] Error in resolution data: {resolution_data['error']}")
                return self._create_error_explainability_result(resolution_data['error'])
            
            # Extract query
            if not query:
                query = resolution_data.get("query", "Unknown query")
            
            # Get search results from resolution data
            search_results = resolution_data.get("search_results", [])
            
            # Convert to evidence tickets
            evidence_tickets = []
            for result in search_results:
                evidence_ticket = EvidenceTicket(
                    ticket_id=result.get("log_id", "unknown"),
                    title=f"Evidence from {result.get('log_id', 'unknown')}",
                    similarity_score=result.get("similarity_score", 0.0),
                    confidence_percentage=self._calculate_confidence_percentage(result.get("similarity_score", 0.0)),
                    source_type="log",
                    resolution=result.get("metadata", {}).get("resolution"),
                    category=result.get("metadata", {}).get("category"),
                    severity=result.get("metadata", {}).get("severity"),
                    environment=result.get("metadata", {}).get("environment"),
                    service=result.get("metadata", {}).get("service"),
                    created_at=result.get("metadata", {}).get("created_at"),
                    metadata=result.get("metadata", {})
                )
                evidence_tickets.append(evidence_ticket)
            
            # Analyze common fixes
            common_fixes = self._analyze_common_fixes(evidence_tickets)
            
            # Calculate overall confidence
            if evidence_tickets:
                overall_confidence = int(np.mean([t.confidence_percentage for t in evidence_tickets[:5]]))
            else:
                overall_confidence = 0
            
            # Generate reasoning and recommendations
            reasoning = self._generate_reasoning(query, evidence_tickets, common_fixes)
            recommendations = self._generate_recommendations(evidence_tickets, common_fixes)
            
            # Calculate transparency score
            transparency_score = self._calculate_transparency_score(evidence_tickets, common_fixes)
            
            # Create result
            result = ExplainabilityResult(
                query=query,
                overall_confidence=overall_confidence,
                evidence_tickets=evidence_tickets,
                common_fixes=common_fixes,
                reasoning=reasoning,
                recommendations=recommendations,
                transparency_score=transparency_score,
                metadata={
                    "total_evidence_tickets": len(evidence_tickets),
                    "high_confidence_tickets": len([t for t in evidence_tickets if t.confidence_percentage >= 80]),
                    "medium_confidence_tickets": len([t for t in evidence_tickets if 60 <= t.confidence_percentage < 80]),
                    "low_confidence_tickets": len([t for t in evidence_tickets if t.confidence_percentage < 60]),
                    "resolved_tickets": len([t for t in evidence_tickets if t.resolution]),
                    "analysis_timestamp": datetime.now().isoformat(),
                    "resolution_data_integrated": True,
                    "recommendations_count": resolution_data.get("recommendations_count", 0),
                    "has_action_plan": resolution_data.get("has_action_plan", False)
                }
            )
            
            print(f"[process_from_resolution_agent] Explainability analysis complete. Overall confidence: {overall_confidence}%")
            return result
            
        except Exception as e:
            print(f"[process_from_resolution_agent] Error: {e}")
            return self._create_error_explainability_result(str(e))
    
    def _create_error_explainability_result(self, error_message: str) -> ExplainabilityResult:
        """Create an explainability result for error cases"""
        return ExplainabilityResult(
            query="Error case",
            overall_confidence=0,
            evidence_tickets=[],
            common_fixes=[],
            reasoning=f"Error occurred during analysis: {error_message}",
            recommendations=["Please check the input data and try again"],
            transparency_score=0.0,
            metadata={
                "error": True,
                "error_message": error_message,
                "analysis_timestamp": datetime.now().isoformat()
            }
        )
    
    def prepare_for_multimodal_agent(self, 
                                   explainability_result: ExplainabilityResult) -> Dict[str, Any]:
        """
        Prepare explainability result for multimodal agent processing.
        This method formats the explainability data in a way that's optimized for multimodal analysis.
        
        Args:
            explainability_result: ExplainabilityResult from explain_incident or process_from_resolution_agent
            
        Returns:
            Dictionary containing data optimized for multimodal agent
        """
        try:
            # Extract key information for multimodal analysis
            multimodal_data = {
                "query": explainability_result.query,
                "overall_confidence": explainability_result.overall_confidence,
                "transparency_score": explainability_result.transparency_score,
                "evidence_summary": {
                    "total_tickets": len(explainability_result.evidence_tickets),
                    "high_confidence_count": len([t for t in explainability_result.evidence_tickets if t.confidence_percentage >= 80]),
                    "resolved_count": len([t for t in explainability_result.evidence_tickets if t.resolution])
                },
                "common_fixes_summary": [
                    {
                        "description": fix["fix_description"][:100] + "..." if len(fix["fix_description"]) > 100 else fix["fix_description"],
                        "frequency": fix["frequency"],
                        "percentage": fix["percentage"]
                    }
                    for fix in explainability_result.common_fixes[:3]
                ],
                "recommendations": explainability_result.recommendations,
                "reasoning": explainability_result.reasoning,
                "metadata": explainability_result.metadata,
                "multimodal_analysis_ready": True,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            print(f"[prepare_for_multimodal_agent] Prepared multimodal data for query: {explainability_result.query}")
            return multimodal_data
            
        except Exception as e:
            print(f"[prepare_for_multimodal_agent] Error: {e}")
            return {"error": str(e), "multimodal_analysis_ready": False}

    def format_explanation(self, result: ExplainabilityResult) -> str:
        """Format the explainability result into a human-readable string."""
        output = []
        
        # Header
        output.append("=" * 80)
        output.append("INCIDENT EXPLAINABILITY ANALYSIS")
        output.append("=" * 80)
        output.append(f"Query: {result.query}")
        output.append(f"Overall Confidence: {result.overall_confidence}%")
        output.append(f"Transparency Score: {result.transparency_score:.1f}/100")
        output.append("")
        
        # Evidence tickets
        output.append("EVIDENCE TICKETS:")
        output.append("-" * 40)
        for i, ticket in enumerate(result.evidence_tickets[:5], 1):
            output.append(f"{i}. Ticket #{ticket.ticket_id[:8]}... - {ticket.confidence_percentage}% similar")
            output.append(f"   Title: {ticket.title}")
            if ticket.category:
                output.append(f"   Category: {ticket.category}")
            if ticket.severity:
                output.append(f"   Severity: {ticket.severity}")
            if ticket.environment:
                output.append(f"   Environment: {ticket.environment}")
            if ticket.resolution:
                output.append(f"   Resolution: {ticket.resolution[:100]}{'...' if len(ticket.resolution) > 100 else ''}")
            output.append("")
        
        # Common fixes
        if result.common_fixes:
            output.append("COMMON FIXES:")
            output.append("-" * 40)
            for i, fix in enumerate(result.common_fixes[:3], 1):
                output.append(f"{i}. {fix['fix_description']}")
                output.append(f"   Used in {fix['frequency']} tickets ({fix['percentage']}%)")
                output.append("")
        
        # Recommendations
        output.append("RECOMMENDATIONS:")
        output.append("-" * 40)
        for i, rec in enumerate(result.recommendations, 1):
            output.append(f"{i}. {rec}")
        output.append("")
        
        # Reasoning
        output.append("DETAILED REASONING:")
        output.append("-" * 40)
        output.append(result.reasoning)
        
        return "\n".join(output)

    def get_explainability_stats(self) -> Dict[str, Any]:
        """Get statistics about the explainability system."""
        try:
            table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            
            stats_query = f"""
            SELECT 
                COUNT(*) as total_incidents,
                COUNTIF(resolution IS NOT NULL AND resolution != '') as resolved_incidents,
                COUNTIF(severity = 'critical') as critical_incidents,
                COUNTIF(severity = 'error') as error_incidents,
                COUNTIF(environment = 'prod') as prod_incidents,
                COUNTIF(environment = 'staging') as staging_incidents,
                COUNTIF(environment = 'dev') as dev_incidents
            FROM `{table_id}`
            """
            
            query_job = self.bq_client.query(stats_query)
            results = list(query_job.result())
            
            if not results:
                return {"error": "No data found in BigQuery table"}
            
            row = results[0]
            
            return {
                "total_incidents": row.total_incidents,
                "resolved_incidents": row.resolved_incidents,
                "resolution_rate": round((row.resolved_incidents / row.total_incidents) * 100, 2) if row.total_incidents > 0 else 0,
                "critical_incidents": row.critical_incidents,
                "error_incidents": row.error_incidents,
                "by_environment": {
                    "production": row.prod_incidents,
                    "staging": row.staging_incidents,
                    "development": row.dev_incidents
                },
                "explainability_available": True
            }
            
        except Exception as e:
            print(f"[get_explainability_stats] error: {e}")
            return {
                "error": str(e),
                "explainability_available": False
            }


def create_explainability_agent(project_id: str = None, 
                              dataset_id: str = "incident_data", 
                              table_id: str = "incidents") -> ExplainabilityAgent:
    """Create and initialize an ExplainabilityAgent instance."""
    if project_id is None:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
    
    return ExplainabilityAgent(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id
    )


def explain_incident(agent: ExplainabilityAgent, query: str, k: int = 10, **filters) -> ExplainabilityResult:
    """Explain an incident using the explainability agent."""
    return agent.explain_incident(query, k=k, filter_metadata=filters if filters else None)


def ensure_explainability_agent_ready(agent: ExplainabilityAgent) -> bool:
    """Ensure the explainability agent is ready for the workflow.
    
    Args:
        agent: ExplainabilityAgent instance
        
    Returns:
        True if explainability agent is ready, False otherwise
    """
    try:
        # Test the agent with a simple query
        test_result = agent.explain_incident("test explainability functionality", k=1)
        
        if test_result and test_result.overall_confidence >= 0:
            print(f"[ensure_explainability_agent_ready] Explainability agent ready")
            return True
        else:
            print(f"[ensure_explainability_agent_ready] Explainability agent not responding properly")
            return False
        
    except Exception as e:
        print(f"[ensure_explainability_agent_ready] Error: {e}")
        return False


def main():
    """Main function to demonstrate explainability agent functionality."""
    # Initialize the explainability agent
    agent = create_explainability_agent()
    
    # Test queries
    test_queries = [
        "authentication error 401 users cannot login auth service",
        "database connection timeout pod crashes",
        "memory leak service restarts",
        "network latency spike performance issues"
    ]
    
    print("EXPLAINABILITY AGENT DEMONSTRATION")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nAnalyzing: '{query}'")
        print("-" * 50)
        
        # Get explainability result
        result = agent.explain_incident(query, k=5)
        
        # Format and display
        formatted_output = agent.format_explanation(result)
        print(formatted_output)
        print("\n" + "=" * 80 + "\n")
    
    # Get system statistics
    print("SYSTEM STATISTICS:")
    print("-" * 30)
    stats = agent.get_explainability_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
