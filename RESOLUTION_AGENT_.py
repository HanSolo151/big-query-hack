"""
Resolution Agent - Uses Gemini to generate actionable recommendations and action plans
Summarizes retrieved solutions and produces concise, actionable recommendations
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage            
from langchain.prompts import ChatPromptTemplate

# Import SearchResult from SearchAgent
from SEARCH_AGENT_1 import SearchResult, LogData                                                       


@dataclass
class ResolutionRecommendation:
    """Data class for resolution recommendations"""
    title: str
    description: str
    priority: str  # High, Medium, Low
    confidence_score: float  # 0.0 to 1.0
    estimated_time: str  # e.g., "5 minutes", "1 hour"
    steps: List[str]
    prerequisites: List[str]
    related_artifacts: List[str]  # log_ids, incident_ids, etc.


@dataclass
class ActionPlan:
    """Data class for multi-step action plans"""
    plan_id: str
    title: str
    description: str
    total_estimated_time: str
    steps: List[Dict[str, Any]]  # Each step has: order, action, description, estimated_time, dependencies
    success_criteria: List[str]
    rollback_plan: List[str]


class ResolutionAgent:
    """
    Resolution Agent that uses Gemini to generate actionable recommendations
    and multi-step action plans from retrieved solutions
    """
    
    def __init__(self, 
                 api_key_path: str = "Gemini_API_Key.txt",
                 model_name: str = "gemini-1.5-flash"):
        """
        Initialize the ResolutionAgent
                                                                         
        Args:
            api_key_path: Path to Gemini API key
            model_name: Gemini model to use
        """
        self.api_key_path = api_key_path
        self.model_name = model_name
        
        # Initialize Gemini
        self._setup_gemini()
        
        # Initialize prompt templates                  
        self._setup_prompts()
        
    def _setup_gemini(self):
        """Set up Gemini text generator"""
        try:
            # Read API key
            with open(self.api_key_path, 'r') as f:
                api_key = f.read().strip()
            
            # Initialize Gemini
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=api_key,
                temperature=0.3,  # Lower temperature for more consistent outputs
                max_output_tokens=2048
            )
            
            print("✓ Resolution Agent Gemini model configured successfully")
            
        except Exception as e:
            print(f"✗ Resolution Agent setup failed: {e}")
            raise
    
    def _setup_prompts(self):
        """Set up prompt templates for different resolution tasks"""
        
        # Template for summarizing solutions
        self.summarize_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a DevOps expert resolution agent. Your task is to analyze retrieved incident data and generate concise, actionable recommendations.

Given incident data, you should:
1. Identify the root cause
2. Provide specific, actionable steps
3. Estimate time and priority
4. Include prerequisites and dependencies

Be concise, technical, and practical. Focus on immediate actions that can resolve the issue."""),
            HumanMessage(content="""Analyze the following incident data and provide resolution recommendations:

{incident_data}

Provide your response in this JSON format:
{{
    "root_cause": "Brief description of the root cause",
    "recommendations": [
        {{
            "title": "Action title",
            "description": "Detailed action description",
            "priority": "High/Medium/Low",
            "confidence": 0.0-1.0,
            "estimated_time": "X minutes/hours",
            "steps": ["step1", "step2", "step3"],
            "prerequisites": ["prereq1", "prereq2"],
            "related_artifacts": ["log_id1", "incident_id1"]
        }}
    ]
}}""")
        ])
        
        # Template for multi-step action plans
        self.action_plan_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a DevOps expert creating comprehensive action plans. Your task is to create detailed, sequential action plans for complex incidents.

For complex issues requiring multiple steps, create:
1. Sequential steps with dependencies
2. Time estimates for each step
3. Success criteria
4. Rollback procedures
5. Risk assessment

Be thorough but practical. Consider dependencies between steps."""),
            HumanMessage(content="""Create a comprehensive action plan for this complex incident:

{incident_data}

Provide your response in this JSON format:
{{
    "plan_title": "Action Plan Title",
    "description": "Overall plan description",
    "total_estimated_time": "X hours",
    "steps": [
        {{
            "order": 1,
            "action": "Action name",
            "description": "Detailed description",
            "estimated_time": "X minutes",
            "dependencies": ["step1", "step2"],
            "risk_level": "Low/Medium/High"
        }}
    ],
    "success_criteria": ["criteria1", "criteria2"],
    "rollback_plan": ["rollback_step1", "rollback_step2"]
}}""")
        ])
    
    def summarize_solutions(self, 
                          search_results: List[SearchResult],
                          query: str = "") -> List[ResolutionRecommendation]:
        """
        Summarize retrieved solutions and generate actionable recommendations
        
        Args:
            search_results: List of SearchResult objects from vector search
            query: Original search query for context
            
        Returns:
            List of ResolutionRecommendation objects
        """
        try:
            if not search_results:
                return []
            
            # Prepare incident data for analysis
            incident_data = self._prepare_incident_data(search_results, query)
            
            # Generate recommendations using Gemini
            response = self.summarize_prompt.format_messages(incident_data=incident_data)
            llm_response = self.llm.invoke(response)
            
            # Parse JSON response
            recommendations_data = json.loads(llm_response.content)
            
            # Convert to ResolutionRecommendation objects
            recommendations = []
            for rec_data in recommendations_data.get("recommendations", []):
                recommendation = ResolutionRecommendation(
                    title=rec_data.get("title", ""),
                    description=rec_data.get("description", ""),
                    priority=rec_data.get("priority", "Medium"),
                    confidence_score=rec_data.get("confidence", 0.5),
                    estimated_time=rec_data.get("estimated_time", "Unknown"),
                    steps=rec_data.get("steps", []),
                    prerequisites=rec_data.get("prerequisites", []),
                    related_artifacts=rec_data.get("related_artifacts", [])
                )
                recommendations.append(recommendation)
            
            print(f"✓ Generated {len(recommendations)} resolution recommendations")
            return recommendations
            
        except Exception as e:
            print(f"✗ Failed to summarize solutions: {e}")
            return []
    
<<<<<<< HEAD
=======
    def process_for_explainability_agent(self, 
                                       search_results: List[SearchResult],
                                       query: str = "") -> Dict[str, Any]:
        """
        Process search results and prepare data for explainability agent.
        This method formats the data in a way that's optimized for explainability analysis.
        
        Args:
            search_results: List of SearchResult objects from vector search
            query: Original search query for context
            
        Returns:
            Dictionary containing processed data for explainability agent
        """
        try:
            if not search_results:
                return {"error": "No search results provided"}
            
            # Generate recommendations first
            recommendations = self.summarize_solutions(search_results, query)
            
            # Create action plan if complex enough
            action_plan = self.create_action_plan(search_results, query)
            
            # Prepare explainability data
            explainability_data = {
                "query": query,
                "search_results_count": len(search_results),
                "recommendations_count": len(recommendations),
                "has_action_plan": action_plan is not None,
                "search_results": [
                    {
                        "log_id": result.log_id,
                        "content": result.content,
                        "similarity_score": result.similarity_score,
                        "metadata": result.metadata,
                        "has_resolution": bool(result.metadata.get('resolution') or 
                                             'resolution' in result.content.lower() or
                                             'fix' in result.content.lower())
                    }
                    for result in search_results
                ],
                "recommendations": [
                    {
                        "title": rec.title,
                        "priority": rec.priority,
                        "confidence_score": rec.confidence_score,
                        "estimated_time": rec.estimated_time,
                        "steps_count": len(rec.steps)
                    }
                    for rec in recommendations
                ],
                "action_plan": {
                    "title": action_plan.title if action_plan else None,
                    "description": action_plan.description if action_plan else None,
                    "total_estimated_time": action_plan.total_estimated_time if action_plan else None,
                    "steps_count": len(action_plan.steps) if action_plan else 0
                } if action_plan else None,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            print(f"✓ Prepared explainability data for {len(search_results)} search results")
            return explainability_data
            
        except Exception as e:
            print(f"✗ Failed to process for explainability agent: {e}")
            return {"error": str(e)}
    
>>>>>>> 1191854 (agentic system)
    def create_action_plan(self, 
                          search_results: List[SearchResult],
                          query: str = "",
                          complexity_threshold: int = 3) -> Optional[ActionPlan]:
        """
        Create a multi-step action plan for complex incidents
        
        Args:
            search_results: List of SearchResult objects
            query: Original search query
            complexity_threshold: Minimum number of recommendations to trigger action plan
            
        Returns:
            ActionPlan object or None if not complex enough
        """
        try:
            # First get recommendations
            recommendations = self.summarize_solutions(search_results, query)
            
            # Only create action plan if complex enough
            if len(recommendations) < complexity_threshold:
                return None
            
            # Prepare incident data for action planning
            incident_data = self._prepare_incident_data(search_results, query)
            
            # Generate action plan using Gemini
            response = self.action_plan_prompt.format_messages(incident_data=incident_data)
            llm_response = self.llm.invoke(response)
            
            # Parse JSON response
            plan_data = json.loads(llm_response.content)
            
            # Create ActionPlan object
            action_plan = ActionPlan(
                plan_id=f"PLAN-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                title=plan_data.get("plan_title", "Action Plan"),
                description=plan_data.get("description", ""),
                total_estimated_time=plan_data.get("total_estimated_time", "Unknown"),
                steps=plan_data.get("steps", []),
                success_criteria=plan_data.get("success_criteria", []),
                rollback_plan=plan_data.get("rollback_plan", [])
            )
            
            print(f"✓ Created action plan: {action_plan.title}")
            return action_plan
            
        except Exception as e:
            print(f"✗ Failed to create action plan: {e}")
            return None
    
    def _prepare_incident_data(self, 
                              search_results: List[SearchResult], 
                              query: str) -> str:
        """Prepare incident data for Gemini analysis"""
        try:
            data_parts = []
            
            if query:
                data_parts.append(f"SEARCH QUERY: {query}")
                data_parts.append("")
            
            data_parts.append("RETRIEVED INCIDENT DATA:")
            data_parts.append("=" * 50)
            
            for i, result in enumerate(search_results, 1):
                data_parts.append(f"\n{i}. INCIDENT {result.log_id}")
                data_parts.append(f"   Similarity Score: {result.similarity_score:.3f}")
                data_parts.append(f"   Content: {result.content}")
                data_parts.append(f"   Metadata: {json.dumps(result.metadata, indent=2)}")
                
                # Include related context if available
                if result.related_context:
                    data_parts.append(f"   Related Context: {json.dumps(result.related_context, indent=2)}")
                
                data_parts.append("-" * 30)
            
            return "\n".join(data_parts)
            
        except Exception as e:
            print(f"✗ Failed to prepare incident data: {e}")
            return f"Error preparing data: {e}"
    
    def generate_quick_fix(self, 
                          search_results: List[SearchResult]) -> Optional[str]:
        """
        Generate a quick fix recommendation for simple issues
        
        Args:
            search_results: List of SearchResult objects
            
        Returns:
            Quick fix string or None
        """
        try:
            if not search_results:
                return None
            
            # Use the most similar result
            top_result = max(search_results, key=lambda x: x.similarity_score)
            
            quick_prompt = f"""
            Based on this incident data, provide a quick fix recommendation in 1-2 sentences:
            
            {top_result.content}
            
            Metadata: {top_result.metadata}
            
            Quick fix:
            """
            
            response = self.llm.invoke([HumanMessage(content=quick_prompt)])
            return response.content.strip()
            
        except Exception as e:
            print(f"✗ Failed to generate quick fix: {e}")
            return None
    
    def analyze_patterns(self, 
                        search_results: List[SearchResult]) -> Dict[str, Any]:
        """
        Analyze patterns across multiple incidents
        
        Args:
            search_results: List of SearchResult objects
            
        Returns:
            Dictionary with pattern analysis
        """
        try:
            if len(search_results) < 2:
                return {"message": "Need at least 2 incidents for pattern analysis"}
            
            # Prepare data for pattern analysis
            incident_data = self._prepare_incident_data(search_results, "")
            
            pattern_prompt = f"""
            Analyze patterns across these incidents and identify:
            1. Common root causes
            2. Recurring issues
            3. Service/environment correlations
            4. Time-based patterns
            5. Recommended preventive measures
            
            Incident Data:
            {incident_data}
            
            Provide analysis in JSON format:
            {{
                "common_causes": ["cause1", "cause2"],
                "recurring_issues": ["issue1", "issue2"],
                "service_correlations": {{"service1": "issue1", "service2": "issue2"}},
                "environment_correlations": {{"prod": "issue1", "staging": "issue2"}},
                "preventive_measures": ["measure1", "measure2"],
                "risk_assessment": "High/Medium/Low"
            }}
            """
            
            response = self.llm.invoke([HumanMessage(content=pattern_prompt)])
            patterns = json.loads(response.content)
            
            print("✓ Pattern analysis completed")
            return patterns
            
        except Exception as e:
            print(f"✗ Failed to analyze patterns: {e}")
            return {"error": str(e)}


<<<<<<< HEAD
=======
def ensure_resolution_agent_ready(agent: ResolutionAgent) -> bool:
    """Ensure the resolution agent is ready for the workflow.
    
    Args:
        agent: ResolutionAgent instance
        
    Returns:
        True if resolution agent is ready, False otherwise
    """
    try:
        # Test the agent with a simple query
        test_prompt = "Test resolution agent functionality"
        response = agent.llm.invoke([HumanMessage(content=test_prompt)])
        
        if response and response.content:
            print(f"[ensure_resolution_agent_ready] Resolution agent ready")
            return True
        else:
            print(f"[ensure_resolution_agent_ready] Resolution agent not responding")
            return False
        
    except Exception as e:
        print(f"[ensure_resolution_agent_ready] Error: {e}")
        return False


>>>>>>> 1191854 (agentic system)
def main():
    """
    Main function demonstrating the ResolutionAgent usage
    """
    try:
        # Initialize the resolution agent
        print("🚀 Initializing Resolution Agent...")
        agent = ResolutionAgent()
        
        # Sample search results (in real scenario, these would come from SearchAgent)
        sample_results = [
            SearchResult(
                log_id="LOG-001",
                content="Database connection timeout issues affecting application performance. Observed high packet drop between nodes.",
                similarity_score=0.95,
                metadata={
                    "service": "payment-service",
                    "environment": "prod",
                    "severity": "error",
                    "priority": "High",
                    "status": "open",
                    "cluster": "cluster-a",
                    "namespace": "auth"
                },
                related_context={
                    "related_logs": [
                        {"log_id": "LOG-002", "severity": "error", "title": "DB Connection Pool Exhausted"}
                    ],
                    "incident": [
                        {"incident_id": "INC-001", "status": "open", "resolution": None}
                    ]
                }
            ),
            SearchResult(
                log_id="LOG-003",
                content="Memory leak detected in order-service. Persistent volume reached 100% capacity.",
                similarity_score=0.87,
                metadata={
                    "service": "order-service",
                    "environment": "staging",
                    "severity": "warn",
                    "priority": "Medium",
                    "status": "resolved",
                    "cluster": "cluster-c",
                    "namespace": "orders"
                },
                related_context={
                    "related_logs": [
                        {"log_id": "LOG-004", "severity": "warn", "title": "High Memory Usage"}
                    ]
                }
            )
        ]
        
        # Generate recommendations
        print("\n📋 Generating resolution recommendations...")
        recommendations = agent.summarize_solutions(sample_results, "database performance issues")
        
        print(f"\n🎯 Resolution Recommendations:")
        print("=" * 50)
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec.title}")
            print(f"   Priority: {rec.priority}")
            print(f"   Confidence: {rec.confidence_score:.2f}")
            print(f"   Estimated Time: {rec.estimated_time}")
            print(f"   Description: {rec.description}")
            print(f"   Steps: {', '.join(rec.steps)}")
            if rec.prerequisites:
                print(f"   Prerequisites: {', '.join(rec.prerequisites)}")
        
        # Create action plan for complex issues
        print(f"\n📋 Creating action plan...")
        action_plan = agent.create_action_plan(sample_results, "database performance issues")
        
        if action_plan:
            print(f"\n🎯 Action Plan: {action_plan.title}")
            print("=" * 50)
            print(f"Description: {action_plan.description}")
            print(f"Total Estimated Time: {action_plan.total_estimated_time}")
            
            print(f"\nSteps:")
            for step in action_plan.steps:
                print(f"  {step.get('order', '?')}. {step.get('action', 'Unknown Action')}")
                print(f"     Time: {step.get('estimated_time', 'Unknown')}")
                print(f"     Risk: {step.get('risk_level', 'Unknown')}")
                if step.get('dependencies'):
                    print(f"     Dependencies: {', '.join(step['dependencies'])}")
            
            print(f"\nSuccess Criteria:")
            for criteria in action_plan.success_criteria:
                print(f"  ✓ {criteria}")
            
            if action_plan.rollback_plan:
                print(f"\nRollback Plan:")
                for rollback in action_plan.rollback_plan:
                    print(f"  ↶ {rollback}")
        
        # Generate quick fix
        print(f"\n⚡ Quick Fix:")
        quick_fix = agent.generate_quick_fix(sample_results)
        if quick_fix:
            print(f"  {quick_fix}")
        
        # Analyze patterns
        print(f"\n🔍 Pattern Analysis:")
        patterns = agent.analyze_patterns(sample_results)
        for key, value in patterns.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Resolution Agent demonstration completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
