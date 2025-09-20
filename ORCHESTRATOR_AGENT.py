"""
Orchestrator Agent - Main Workflow Coordinator
Coordinates the entire incident management workflow by connecting all agents in sequence:
1. Embedding Agent → 2. Search Agent → 3. Resolution Agent → 4. Explainability Agent → 5. Multimodal Agent → 6. Feedback Integration Agent
Proactive Agent runs in parallel for continuous monitoring
"""

import os
import json
import asyncio
import threading
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

# Import all agents
from EMBEDDING_AGENT_1 import EmbeddingAgent, LogData, create_agent as create_embedding_agent
from SEARCH_AGENT_1 import VectorSearchAgent, SearchResult
from RESOLUTION_AGENT_ import ResolutionAgent, ResolutionRecommendation, ActionPlan
from EXPLAINABILITY_AGENT_1 import ExplainabilityAgent, ExplainabilityResult, create_explainability_agent
from MULTIMODAL_AGENT_1 import MultimodalAgent, MultimodalSearchResult, create_multimodal_agent
from FEEDBACK_INTEGRATION_AGENT import FeedbackIntegrationAgent, UserFeedback
from PROACTIVE_AGENT import ProactiveAgent, ProactiveAlert, EmergingIssue

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WorkflowResult:
    """Complete workflow result containing outputs from all agents"""
    session_id: str
    query: str
    timestamp: datetime
    
    # Agent outputs
    search_results: List[SearchResult]
    resolution_recommendations: List[ResolutionRecommendation]
    action_plan: Optional[ActionPlan]
    explainability_result: Optional[ExplainabilityResult]
    multimodal_results: List[MultimodalSearchResult]
    feedback_collected: bool
    
    # Workflow metadata
    total_processing_time: float
    agent_processing_times: Dict[str, float]
    success: bool
    error_message: Optional[str] = None

@dataclass
class SystemStatus:
    """System status and health metrics"""
    timestamp: datetime
    agents_status: Dict[str, bool]
    active_sessions: int
    total_queries_processed: int
    average_processing_time: float
    proactive_alerts: int
    emerging_issues: int
    system_health: str  # "healthy", "degraded", "critical"

class OrchestratorAgent:
    """
    Main Orchestrator Agent that coordinates the entire incident management workflow
    """
    
    def __init__(self, 
                 project_id: str = "big-station-472112-i1",
                 credentials_path: str = "big-station-472112-i1-01b16573569e.json",
                 api_key_path: str = "Gemini_API_Key.txt"):
        """
        Initialize the Orchestrator Agent
        
        Args:
            project_id: Google Cloud project ID
            credentials_path: Path to service account credentials
            api_key_path: Path to Gemini API key
        """
        self.project_id = project_id
        self.credentials_path = credentials_path
        self.api_key_path = api_key_path
        
        # Initialize all agents
        self._initialize_agents()
        
        # Workflow state
        self.active_sessions: Dict[str, WorkflowResult] = {}
        self.total_queries_processed = 0
        self.proactive_agent_running = False
        
        # Performance tracking
        self.processing_times: List[float] = []
        
        logger.info("✓ Orchestrator Agent initialized successfully")
    
    def _initialize_agents(self):
        """Initialize all agents in the workflow"""
        try:
            # 1. Embedding Agent (connects to BigQuery, creates embeddings, stores in vector store)
            logger.info("Initializing Embedding Agent...")
            self.embedding_agent = create_embedding_agent(project_id=self.project_id)
            
            # 2. Search Agent (uses vector store, executes vector search in BigQuery/BigFrames)
            logger.info("Initializing Search Agent...")
            self.search_agent = VectorSearchAgent(
                project_id=self.project_id,
                credentials_path=self.credentials_path,
                api_key_path=self.api_key_path
            )
            
            # 3. Resolution Agent (uses GeminiTextGenerator to summarize retrieved fixes)
            logger.info("Initializing Resolution Agent...")
            self.resolution_agent = ResolutionAgent(api_key_path=self.api_key_path)
            
            # 4. Explainability Agent (returns confidence scores + supporting incidents)
            logger.info("Initializing Explainability Agent...")
            self.explainability_agent = create_explainability_agent(project_id=self.project_id)
            
            # 5. Multimodal Agent (connects to GCP for unstructured operational data)
            logger.info("Initializing Multimodal Agent...")
            self.multimodal_agent = create_multimodal_agent(project_id=self.project_id)
            
            # 6. Feedback Integration Agent (outputs to orchestrator)
            logger.info("Initializing Feedback Integration Agent...")
            self.feedback_agent = FeedbackIntegrationAgent(api_key_path=self.api_key_path)
            
            # 7. Proactive Agent (runs in parallel, connects to GCP)
            logger.info("Initializing Proactive Agent...")
            self.proactive_agent = ProactiveAgent(
                project_id=self.project_id,
                credentials_path=self.credentials_path,
                api_key_path=self.api_key_path
            )
            
            logger.info("✓ All agents initialized successfully")
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize agents: {e}")
            raise
    
    def start_proactive_monitoring(self):
        """Start the proactive agent for continuous monitoring"""
        try:
            if not self.proactive_agent_running:
                self.proactive_agent.start_monitoring()
                self.proactive_agent_running = True
                logger.info("✓ Proactive monitoring started")
            else:
                logger.info("! Proactive monitoring already running")
        except Exception as e:
            logger.error(f"✗ Failed to start proactive monitoring: {e}")
            raise
    
    def stop_proactive_monitoring(self):
        """Stop the proactive agent"""
        try:
            if self.proactive_agent_running:
                self.proactive_agent.stop_monitoring()
                self.proactive_agent_running = False
                logger.info("✓ Proactive monitoring stopped")
        except Exception as e:
            logger.error(f"✗ Failed to stop proactive monitoring: {e}")
    
    async def process_incident_query(self, 
                                   query: str, 
                                   user_id: str = "system",
                                   session_id: str = None,
                                   include_multimodal: bool = True,
                                   k_search: int = 5,
                                   k_explainability: int = 10) -> WorkflowResult:
        """
        Process an incident query through the complete workflow
        
        Args:
            query: Incident description or query
            user_id: ID of the user making the query
            session_id: Session ID for tracking (auto-generated if None)
            include_multimodal: Whether to include multimodal analysis
            k_search: Number of search results to retrieve
            k_explainability: Number of incidents for explainability analysis
            
        Returns:
            WorkflowResult with outputs from all agents
        """
        start_time = time.time()
        
        if session_id is None:
            session_id = f"SESSION-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{user_id[:8]}"
        
        logger.info(f"Processing incident query: '{query}' (Session: {session_id})")
        
        # Initialize workflow result
        workflow_result = WorkflowResult(
            session_id=session_id,
            query=query,
            timestamp=datetime.now(),
            search_results=[],
            resolution_recommendations=[],
            action_plan=None,
            explainability_result=None,
            multimodal_results=[],
            feedback_collected=False,
            total_processing_time=0.0,
            agent_processing_times={},
            success=False
        )
        
        try:
            # Step 1: Search Agent - Vector search for similar incidents
            logger.info("Step 1: Executing vector search...")
            search_start = time.time()
            
            search_results = self.search_agent.vector_search(
                query=query,
                k=k_search
            )
            
            search_time = time.time() - search_start
            workflow_result.agent_processing_times['search'] = search_time
            workflow_result.search_results = search_results
            
            logger.info(f"✓ Found {len(search_results)} similar incidents")
            
            # Step 2: Resolution Agent - Generate actionable recommendations
            logger.info("Step 2: Generating resolution recommendations...")
            resolution_start = time.time()
            
            resolution_recommendations = self.resolution_agent.summarize_solutions(
                search_results=search_results,
                query=query
            )
            
            # Create action plan for complex issues
            action_plan = self.resolution_agent.create_action_plan(
                search_results=search_results,
                query=query
            )
            
            resolution_time = time.time() - resolution_start
            workflow_result.agent_processing_times['resolution'] = resolution_time
            workflow_result.resolution_recommendations = resolution_recommendations
            workflow_result.action_plan = action_plan
            
            logger.info(f"✓ Generated {len(resolution_recommendations)} recommendations")
            if action_plan:
                logger.info(f"✓ Created action plan: {action_plan.title}")
            
            # Step 3: Explainability Agent - Provide confidence scores and evidence
            logger.info("Step 3: Analyzing explainability...")
            explainability_start = time.time()
            
            explainability_result = self.explainability_agent.explain_incident(
                query=query,
                k=k_explainability
            )
            
            explainability_time = time.time() - explainability_start
            workflow_result.agent_processing_times['explainability'] = explainability_time
            workflow_result.explainability_result = explainability_result
            
            logger.info(f"✓ Explainability analysis complete (Confidence: {explainability_result.overall_confidence}%)")
            
            # Step 4: Multimodal Agent - Analyze unstructured data (if enabled)
            if include_multimodal:
                logger.info("Step 4: Analyzing multimodal data...")
                multimodal_start = time.time()
                
                multimodal_results = self.multimodal_agent.search_similar_multimodal(
                    query=query,
                    k=k_search
                )
                
                multimodal_time = time.time() - multimodal_start
                workflow_result.agent_processing_times['multimodal'] = multimodal_time
                workflow_result.multimodal_results = multimodal_results
                
                logger.info(f"✓ Found {len(multimodal_results)} multimodal matches")
            
            # Step 5: Feedback Integration Agent - Collect feedback for learning
            logger.info("Step 5: Setting up feedback collection...")
            feedback_start = time.time()
            
            # The feedback agent is ready to collect feedback
            # In a real implementation, this would be triggered by user interaction
            workflow_result.feedback_collected = True
            
            feedback_time = time.time() - feedback_start
            workflow_result.agent_processing_times['feedback'] = feedback_time
            
            # Calculate total processing time
            total_time = time.time() - start_time
            workflow_result.total_processing_time = total_time
            workflow_result.success = True
            
            # Store session
            self.active_sessions[session_id] = workflow_result
            self.total_queries_processed += 1
            self.processing_times.append(total_time)
            
            logger.info(f"✓ Workflow completed successfully in {total_time:.2f}s")
            
            return workflow_result
            
        except Exception as e:
            error_msg = f"Workflow failed: {str(e)}"
            logger.error(f"✗ {error_msg}")
            
            workflow_result.success = False
            workflow_result.error_message = error_msg
            workflow_result.total_processing_time = time.time() - start_time
            
            return workflow_result
    
    def collect_user_feedback(self, 
                            session_id: str,
                            user_id: str,
                            feedback_type: str,
                            feedback_value: Union[bool, int, str],
                            feedback_text: Optional[str] = None) -> str:
        """
        Collect user feedback for a session
        
        Args:
            session_id: Session ID to provide feedback for
            user_id: ID of the user providing feedback
            feedback_type: Type of feedback ("thumbs_up", "thumbs_down", "natural_language", "rating")
            feedback_value: Feedback value (bool, int, or str)
            feedback_text: Additional text feedback
            
        Returns:
            Feedback ID for tracking
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            workflow_result = self.active_sessions[session_id]
            
            # Collect feedback using the feedback agent
            feedback_id = self.feedback_agent.collect_feedback(
                user_id=user_id,
                session_id=session_id,
                query=workflow_result.query,
                search_results=workflow_result.search_results,
                recommendations=workflow_result.resolution_recommendations,
                action_plan=workflow_result.action_plan,
                feedback_type=feedback_type,
                feedback_value=feedback_value,
                feedback_text=feedback_text
            )
            
            logger.info(f"✓ Feedback collected: {feedback_id}")
            return feedback_id
            
        except Exception as e:
            logger.error(f"✗ Failed to collect feedback: {e}")
            raise
    
    def get_proactive_alerts(self) -> List[ProactiveAlert]:
        """Get active proactive alerts"""
        try:
            return self.proactive_agent.get_active_alerts()
        except Exception as e:
            logger.error(f"✗ Failed to get proactive alerts: {e}")
            return []
    
    def get_emerging_issues(self) -> List[EmergingIssue]:
        """Get detected emerging issues"""
        try:
            return self.proactive_agent.get_emerging_issues()
        except Exception as e:
            logger.error(f"✗ Failed to get emerging issues: {e}")
            return []
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status and health metrics"""
        try:
            # Check agent status
            agents_status = {
                "embedding_agent": True,  # Assume healthy if initialized
                "search_agent": True,
                "resolution_agent": True,
                "explainability_agent": True,
                "multimodal_agent": True,
                "feedback_agent": True,
                "proactive_agent": self.proactive_agent_running
            }
            
            # Calculate average processing time
            avg_processing_time = 0.0
            if self.processing_times:
                avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            
            # Determine system health
            system_health = "healthy"
            if not self.proactive_agent_running:
                system_health = "degraded"
            if len(self.processing_times) > 0 and avg_processing_time > 30.0:  # > 30 seconds
                system_health = "degraded"
            
            return SystemStatus(
                timestamp=datetime.now(),
                agents_status=agents_status,
                active_sessions=len(self.active_sessions),
                total_queries_processed=self.total_queries_processed,
                average_processing_time=avg_processing_time,
                proactive_alerts=len(self.get_proactive_alerts()),
                emerging_issues=len(self.get_emerging_issues()),
                system_health=system_health
            )
            
        except Exception as e:
            logger.error(f"✗ Failed to get system status: {e}")
            return SystemStatus(
                timestamp=datetime.now(),
                agents_status={},
                active_sessions=0,
                total_queries_processed=0,
                average_processing_time=0.0,
                proactive_alerts=0,
                emerging_issues=0,
                system_health="critical"
            )
    
    def get_session_result(self, session_id: str) -> Optional[WorkflowResult]:
        """Get workflow result for a specific session"""
        return self.active_sessions.get(session_id)
    
    def format_workflow_result(self, result: WorkflowResult) -> str:
        """Format workflow result into a human-readable string"""
        output = []
        
        # Header
        output.append("=" * 80)
        output.append("INCIDENT MANAGEMENT WORKFLOW RESULT")
        output.append("=" * 80)
        output.append(f"Session ID: {result.session_id}")
        output.append(f"Query: {result.query}")
        output.append(f"Timestamp: {result.timestamp}")
        output.append(f"Processing Time: {result.total_processing_time:.2f}s")
        output.append(f"Success: {result.success}")
        output.append("")
        
        if not result.success:
            output.append(f"ERROR: {result.error_message}")
            return "\n".join(output)
        
        # Search Results
        output.append("SEARCH RESULTS:")
        output.append("-" * 40)
        for i, search_result in enumerate(result.search_results, 1):
            output.append(f"{i}. {search_result.log_id} (Score: {search_result.similarity_score:.3f})")
            output.append(f"   Content: {search_result.content[:100]}...")
            output.append("")
        
        # Resolution Recommendations
        output.append("RESOLUTION RECOMMENDATIONS:")
        output.append("-" * 40)
        for i, rec in enumerate(result.resolution_recommendations, 1):
            output.append(f"{i}. {rec.title}")
            output.append(f"   Priority: {rec.priority}")
            output.append(f"   Confidence: {rec.confidence_score:.2f}")
            output.append(f"   Estimated Time: {rec.estimated_time}")
            output.append(f"   Description: {rec.description}")
            output.append("")
        
        # Action Plan
        if result.action_plan:
            output.append("ACTION PLAN:")
            output.append("-" * 40)
            output.append(f"Title: {result.action_plan.title}")
            output.append(f"Description: {result.action_plan.description}")
            output.append(f"Total Time: {result.action_plan.total_estimated_time}")
            output.append("Steps:")
            for step in result.action_plan.steps:
                output.append(f"  {step.get('order', '?')}. {step.get('action', 'Unknown')}")
            output.append("")
        
        # Explainability
        if result.explainability_result:
            output.append("EXPLAINABILITY ANALYSIS:")
            output.append("-" * 40)
            output.append(f"Overall Confidence: {result.explainability_result.overall_confidence}%")
            output.append(f"Transparency Score: {result.explainability_result.transparency_score:.1f}/100")
            output.append(f"Evidence Tickets: {len(result.explainability_result.evidence_tickets)}")
            output.append("")
        
        # Multimodal Results
        if result.multimodal_results:
            output.append("MULTIMODAL ANALYSIS:")
            output.append("-" * 40)
            for i, mm_result in enumerate(result.multimodal_results, 1):
                output.append(f"{i}. {mm_result.title} ({mm_result.content_type})")
                output.append(f"   Confidence: {mm_result.confidence_percentage}%")
                output.append("")
        
        # Performance Metrics
        output.append("PERFORMANCE METRICS:")
        output.append("-" * 40)
        for agent, time_taken in result.agent_processing_times.items():
            output.append(f"{agent}: {time_taken:.2f}s")
        
        return "\n".join(output)


async def main():
    """
    Main function demonstrating the Orchestrator Agent
    """
    try:
        # Initialize the orchestrator
        print("🚀 Initializing Orchestrator Agent...")
        orchestrator = OrchestratorAgent()
        
        # Start proactive monitoring
        print("\n📊 Starting proactive monitoring...")
        orchestrator.start_proactive_monitoring()
        
        # Process sample incident queries
        print("\n🔍 Processing incident queries...")
        
        sample_queries = [
            "database connection timeout errors in production",
            "authentication service 401 errors",
            "payment API timeout issues",
            "memory leak in order processing service"
        ]
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            # Process the query
            result = await orchestrator.process_incident_query(
                query=query,
                user_id=f"user_{i}",
                include_multimodal=True
            )
            
            # Display results
            formatted_result = orchestrator.format_workflow_result(result)
            print(formatted_result)
            
            # Simulate user feedback
            if result.success:
                feedback_id = orchestrator.collect_user_feedback(
                    session_id=result.session_id,
                    user_id=f"user_{i}",
                    feedback_type="thumbs_up",
                    feedback_value=True
                )
                print(f"✓ Feedback collected: {feedback_id}")
        
        # Display system status
        print("\n📊 System Status:")
        status = orchestrator.get_system_status()
        print(f"Active Sessions: {status.active_sessions}")
        print(f"Total Queries Processed: {status.total_queries_processed}")
        print(f"Average Processing Time: {status.average_processing_time:.2f}s")
        print(f"System Health: {status.system_health}")
        print(f"Proactive Alerts: {status.proactive_alerts}")
        print(f"Emerging Issues: {status.emerging_issues}")
        
        # Display proactive alerts
        alerts = orchestrator.get_proactive_alerts()
        if alerts:
            print("\n🚨 Proactive Alerts:")
            for alert in alerts:
                print(f"  {alert.title} ({alert.severity})")
                print(f"    {alert.message}")
        
        # Display emerging issues
        issues = orchestrator.get_emerging_issues()
        if issues:
            print("\n⚠️  Emerging Issues:")
            for issue in issues:
                print(f"  {issue.title} (Confidence: {issue.confidence_score:.2f})")
                print(f"    Growth Rate: {issue.growth_rate:.1f} tickets/hour")
        
        print("\n✅ Orchestrator Agent demonstration completed successfully!")
        print("💡 Complete incident management workflow is now operational!")
        
        # Stop proactive monitoring
        orchestrator.stop_proactive_monitoring()
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
