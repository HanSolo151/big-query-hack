"""
Feedback Integration Agent - Learning Layer
Collects user feedback and continuously improves the system through:
- Embedding weight updates
- Resolution ranking logic improvements  
- Training examples for future queries
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import sqlite3

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

# Import from other agents
from SEARCH_AGENT_1 import SearchResult, VectorSearchAgent
from RESOLUTION_AGENT_ import ResolutionRecommendation, ActionPlan, ResolutionAgent


@dataclass
class UserFeedback:
    """Data class for user feedback"""
    feedback_id: str
    user_id: str
    session_id: str
    query: str
    search_results: List[SearchResult]
    recommendations: List[ResolutionRecommendation]
    action_plan: Optional[ActionPlan]
    
    # Feedback types
    feedback_type: str  # "thumbs_up", "thumbs_down", "natural_language", "rating"
    feedback_value: Union[bool, int, str]  # True/False, 1-5 rating, or text
    feedback_text: Optional[str] = None
    
    # Context
    timestamp: datetime
    resolution_attempted: bool = False
    resolution_successful: bool = False
    time_to_resolution: Optional[timedelta] = None
    
    # Learning signals
    embedding_relevance: float = 0.0  # How relevant were the search results
    recommendation_quality: float = 0.0  # Quality of recommendations
    action_plan_effectiveness: float = 0.0  # Effectiveness of action plan


@dataclass
class LearningMetrics:
    """Data class for tracking learning metrics"""
    total_feedback: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    embedding_improvements: int = 0
    ranking_improvements: int = 0
    training_examples_added: int = 0
    accuracy_trend: List[float] = None
    
    def __post_init__(self):
        if self.accuracy_trend is None:
            self.accuracy_trend = []


class FeedbackIntegrationAgent:
    """
    Feedback Integration Agent that learns from user interactions
    and continuously improves the system
    """
    
    def __init__(self, 
                 db_path: str = "feedback_learning.db",
                 api_key_path: str = "Gemini_API_Key.txt"):
        """
        Initialize the FeedbackIntegrationAgent
        
        Args:
            db_path: Path to SQLite database for storing feedback
            api_key_path: Path to Gemini API key for natural language processing
        """
        self.db_path = db_path
        self.api_key_path = api_key_path
        
        # Initialize components
        self._setup_database()
        self._setup_gemini()
        self._load_learning_state()
        
        # Learning parameters
        self.learning_rate = 0.1
        self.min_feedback_threshold = 5
        self.embedding_weight_decay = 0.95
        
        print("✓ Feedback Integration Agent initialized successfully")
    
    def _setup_database(self):
        """Set up SQLite database for storing feedback and learning data"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # Create feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    session_id TEXT,
                    query TEXT,
                    feedback_type TEXT,
                    feedback_value TEXT,
                    feedback_text TEXT,
                    timestamp TEXT,
                    resolution_attempted BOOLEAN,
                    resolution_successful BOOLEAN,
                    time_to_resolution TEXT,
                    embedding_relevance REAL,
                    recommendation_quality REAL,
                    action_plan_effectiveness REAL,
                    search_results TEXT,
                    recommendations TEXT,
                    action_plan TEXT
                )
            """)
            
            # Create learning metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp TEXT,
                    context TEXT
                )
            """)
            
            # Create embedding weights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_pattern TEXT,
                    weight_vector TEXT,
                    confidence REAL,
                    last_updated TEXT,
                    feedback_count INTEGER
                )
            """)
            
            # Create training examples table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    positive_examples TEXT,
                    negative_examples TEXT,
                    context TEXT,
                    created_at TEXT,
                    last_used TEXT,
                    usage_count INTEGER
                )
            """)
            
            self.conn.commit()
            print("✓ Database setup completed")
            
        except Exception as e:
            print(f"✗ Database setup failed: {e}")
            raise
    
    def _setup_gemini(self):
        """Set up Gemini for natural language feedback processing"""
        try:
            with open(self.api_key_path, 'r') as f:
                api_key = f.read().strip()
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.1
            )
            
            print("✓ Gemini setup completed")
            
        except Exception as e:
            print(f"✗ Gemini setup failed: {e}")
            raise
    
    def _load_learning_state(self):
        """Load existing learning state from database"""
        try:
            cursor = self.conn.cursor()
            
            # Load learning metrics
            cursor.execute("SELECT * FROM learning_metrics ORDER BY timestamp DESC LIMIT 100")
            metrics_data = cursor.fetchall()
            
            self.learning_metrics = LearningMetrics()
            for metric in metrics_data:
                if metric[1] == "accuracy":
                    self.learning_metrics.accuracy_trend.append(metric[2])
            
            # Load embedding weights
            cursor.execute("SELECT * FROM embedding_weights")
            weights_data = cursor.fetchall()
            
            self.embedding_weights = {}
            for weight in weights_data:
                self.embedding_weights[weight[1]] = {
                    'vector': json.loads(weight[2]),
                    'confidence': weight[3],
                    'last_updated': weight[4],
                    'feedback_count': weight[5]
                }
            
            print("✓ Learning state loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load learning state: {e}")
            self.learning_metrics = LearningMetrics()
            self.embedding_weights = {}
    
    def collect_feedback(self,
                        user_id: str,
                        session_id: str,
                        query: str,
                        search_results: List[SearchResult],
                        recommendations: List[ResolutionRecommendation],
                        action_plan: Optional[ActionPlan],
                        feedback_type: str,
                        feedback_value: Union[bool, int, str],
                        feedback_text: Optional[str] = None) -> str:
        """
        Collect user feedback and store it for learning
        
        Args:
            user_id: ID of the user providing feedback
            session_id: Session ID for grouping related interactions
            query: Original search query
            search_results: Results from search agent
            recommendations: Recommendations from resolution agent
            action_plan: Action plan if created
            feedback_type: Type of feedback ("thumbs_up", "thumbs_down", "natural_language", "rating")
            feedback_value: Feedback value (bool, int, or str)
            feedback_text: Additional text feedback
            
        Returns:
            Feedback ID for tracking
        """
        try:
            feedback_id = f"FB-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{user_id[:8]}"
            
            # Create feedback object
            feedback = UserFeedback(
                feedback_id=feedback_id,
                user_id=user_id,
                session_id=session_id,
                query=query,
                search_results=search_results,
                recommendations=recommendations,
                action_plan=action_plan,
                feedback_type=feedback_type,
                feedback_value=feedback_value,
                feedback_text=feedback_text,
                timestamp=datetime.now()
            )
            
            # Process feedback for learning signals
            self._extract_learning_signals(feedback)
            
            # Store in database
            self._store_feedback(feedback)
            
            # Trigger learning updates
            self._trigger_learning_updates(feedback)
            
            print(f"✓ Feedback collected: {feedback_id}")
            return feedback_id
            
        except Exception as e:
            print(f"✗ Failed to collect feedback: {e}")
            raise
    
    def _extract_learning_signals(self, feedback: UserFeedback):
        """Extract learning signals from feedback"""
        try:
            # Process natural language feedback
            if feedback.feedback_type == "natural_language" and feedback.feedback_text:
                signals = self._analyze_natural_language_feedback(feedback.feedback_text)
                feedback.embedding_relevance = signals.get('embedding_relevance', 0.0)
                feedback.recommendation_quality = signals.get('recommendation_quality', 0.0)
                feedback.action_plan_effectiveness = signals.get('action_plan_effectiveness', 0.0)
            
            # Process thumbs up/down feedback
            elif feedback.feedback_type in ["thumbs_up", "thumbs_down"]:
                is_positive = feedback.feedback_type == "thumbs_up"
                feedback.embedding_relevance = 0.8 if is_positive else 0.2
                feedback.recommendation_quality = 0.8 if is_positive else 0.2
                feedback.action_plan_effectiveness = 0.8 if is_positive else 0.2
            
            # Process rating feedback
            elif feedback.feedback_type == "rating":
                rating = float(feedback.feedback_value) / 5.0  # Normalize to 0-1
                feedback.embedding_relevance = rating
                feedback.recommendation_quality = rating
                feedback.action_plan_effectiveness = rating
            
        except Exception as e:
            print(f"✗ Failed to extract learning signals: {e}")
    
    def _analyze_natural_language_feedback(self, feedback_text: str) -> Dict[str, float]:
        """Analyze natural language feedback using Gemini"""
        try:
            prompt = f"""
            Analyze this user feedback and extract learning signals:
            
            Feedback: "{feedback_text}"
            
            Rate the following aspects on a scale of 0.0 to 1.0:
            1. How relevant were the search results? (embedding_relevance)
            2. How helpful were the recommendations? (recommendation_quality)  
            3. How effective was the action plan? (action_plan_effectiveness)
            
            Respond in JSON format:
            {{
                "embedding_relevance": 0.0-1.0,
                "recommendation_quality": 0.0-1.0,
                "action_plan_effectiveness": 0.0-1.0,
                "key_issues": ["issue1", "issue2"],
                "suggestions": ["suggestion1", "suggestion2"]
            }}
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            signals = json.loads(response.content)
            
            return {
                'embedding_relevance': signals.get('embedding_relevance', 0.5),
                'recommendation_quality': signals.get('recommendation_quality', 0.5),
                'action_plan_effectiveness': signals.get('action_plan_effectiveness', 0.5)
            }
            
        except Exception as e:
            print(f"✗ Failed to analyze natural language feedback: {e}")
            return {'embedding_relevance': 0.5, 'recommendation_quality': 0.5, 'action_plan_effectiveness': 0.5}
    
    def _store_feedback(self, feedback: UserFeedback):
        """Store feedback in database"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO feedback (
                    feedback_id, user_id, session_id, query, feedback_type, feedback_value,
                    feedback_text, timestamp, resolution_attempted, resolution_successful,
                    time_to_resolution, embedding_relevance, recommendation_quality,
                    action_plan_effectiveness, search_results, recommendations, action_plan
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.feedback_id,
                feedback.user_id,
                feedback.session_id,
                feedback.query,
                feedback.feedback_type,
                str(feedback.feedback_value),
                feedback.feedback_text,
                feedback.timestamp.isoformat(),
                feedback.resolution_attempted,
                feedback.resolution_successful,
                str(feedback.time_to_resolution) if feedback.time_to_resolution else None,
                feedback.embedding_relevance,
                feedback.recommendation_quality,
                feedback.action_plan_effectiveness,
                json.dumps([asdict(r) for r in feedback.search_results]),
                json.dumps([asdict(r) for r in feedback.recommendations]),
                json.dumps(asdict(feedback.action_plan)) if feedback.action_plan else None
            ))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"✗ Failed to store feedback: {e}")
            raise
    
    def _trigger_learning_updates(self, feedback: UserFeedback):
        """Trigger learning updates based on feedback"""
        try:
            # Update embedding weights
            self._update_embedding_weights(feedback)
            
            # Update resolution ranking logic
            self._update_resolution_ranking(feedback)
            
            # Add training examples
            self._add_training_examples(feedback)
            
            # Update learning metrics
            self._update_learning_metrics(feedback)
            
        except Exception as e:
            print(f"✗ Failed to trigger learning updates: {e}")
    
    def _update_embedding_weights(self, feedback: UserFeedback):
        """Update embedding weights based on feedback"""
        try:
            # Extract query patterns
            query_pattern = self._extract_query_pattern(feedback.query)
            
            # Calculate weight adjustment
            relevance_score = feedback.embedding_relevance
            weight_adjustment = (relevance_score - 0.5) * self.learning_rate
            
            # Update or create weight vector
            if query_pattern in self.embedding_weights:
                current_weights = self.embedding_weights[query_pattern]
                new_confidence = (current_weights['confidence'] + relevance_score) / 2
                new_feedback_count = current_weights['feedback_count'] + 1
                
                # Apply weight decay
                current_weights['confidence'] *= self.embedding_weight_decay
                current_weights['confidence'] += weight_adjustment
                current_weights['confidence'] = max(0.0, min(1.0, current_weights['confidence']))
                current_weights['last_updated'] = datetime.now().isoformat()
                current_weights['feedback_count'] = new_feedback_count
            else:
                self.embedding_weights[query_pattern] = {
                    'vector': [weight_adjustment],
                    'confidence': relevance_score,
                    'last_updated': datetime.now().isoformat(),
                    'feedback_count': 1
                }
            
            # Store in database
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO embedding_weights 
                (query_pattern, weight_vector, confidence, last_updated, feedback_count)
                VALUES (?, ?, ?, ?, ?)
            """, (
                query_pattern,
                json.dumps(self.embedding_weights[query_pattern]['vector']),
                self.embedding_weights[query_pattern]['confidence'],
                self.embedding_weights[query_pattern]['last_updated'],
                self.embedding_weights[query_pattern]['feedback_count']
            ))
            
            self.conn.commit()
            self.learning_metrics.embedding_improvements += 1
            
        except Exception as e:
            print(f"✗ Failed to update embedding weights: {e}")
    
    def _update_resolution_ranking(self, feedback: UserFeedback):
        """Update resolution ranking logic based on feedback"""
        try:
            # Analyze which recommendations were most helpful
            if feedback.recommendations:
                for i, rec in enumerate(feedback.recommendations):
                    # Store ranking feedback
                    cursor = self.conn.cursor()
                    cursor.execute("""
                        INSERT INTO learning_metrics (metric_name, metric_value, timestamp, context)
                        VALUES (?, ?, ?, ?)
                    """, (
                        "recommendation_ranking",
                        feedback.recommendation_quality,
                        datetime.now().isoformat(),
                        json.dumps({
                            "recommendation_index": i,
                            "recommendation_title": rec.title,
                            "priority": rec.priority,
                            "confidence": rec.confidence_score
                        })
                    ))
            
            self.conn.commit()
            self.learning_metrics.ranking_improvements += 1
            
        except Exception as e:
            print(f"✗ Failed to update resolution ranking: {e}")
    
    def _add_training_examples(self, feedback: UserFeedback):
        """Add feedback as training examples for future queries"""
        try:
            # Create positive and negative examples
            positive_examples = []
            negative_examples = []
            
            if feedback.embedding_relevance > 0.6:
                positive_examples.extend([asdict(r) for r in feedback.search_results])
            else:
                negative_examples.extend([asdict(r) for r in feedback.search_results])
            
            if feedback.recommendation_quality > 0.6:
                positive_examples.extend([asdict(r) for r in feedback.recommendations])
            else:
                negative_examples.extend([asdict(r) for r in feedback.recommendations])
            
            # Store training examples
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO training_examples 
                (query, positive_examples, negative_examples, context, created_at, last_used, usage_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.query,
                json.dumps(positive_examples),
                json.dumps(negative_examples),
                json.dumps({
                    "feedback_type": feedback.feedback_type,
                    "feedback_value": str(feedback.feedback_value),
                    "session_id": feedback.session_id
                }),
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                0
            ))
            
            self.conn.commit()
            self.learning_metrics.training_examples_added += 1
            
        except Exception as e:
            print(f"✗ Failed to add training examples: {e}")
    
    def _update_learning_metrics(self, feedback: UserFeedback):
        """Update overall learning metrics"""
        try:
            self.learning_metrics.total_feedback += 1
            
            if feedback.embedding_relevance > 0.5:
                self.learning_metrics.positive_feedback += 1
            else:
                self.learning_metrics.negative_feedback += 1
            
            # Calculate accuracy trend
            accuracy = self.learning_metrics.positive_feedback / self.learning_metrics.total_feedback
            self.learning_metrics.accuracy_trend.append(accuracy)
            
            # Store metrics
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO learning_metrics (metric_name, metric_value, timestamp, context)
                VALUES (?, ?, ?, ?)
            """, (
                "accuracy",
                accuracy,
                datetime.now().isoformat(),
                json.dumps({
                    "total_feedback": self.learning_metrics.total_feedback,
                    "positive_feedback": self.learning_metrics.positive_feedback,
                    "negative_feedback": self.learning_metrics.negative_feedback
                })
            ))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"✗ Failed to update learning metrics: {e}")
    
    def _extract_query_pattern(self, query: str) -> str:
        """Extract query pattern for weight tracking"""
        # Simple pattern extraction - could be enhanced with NLP
        words = query.lower().split()
        if len(words) >= 3:
            return f"{words[0]}_{words[1]}_{words[2]}"
        elif len(words) == 2:
            return f"{words[0]}_{words[1]}"
        else:
            return words[0] if words else "unknown"
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the learning progress"""
        try:
            insights = {
                "total_feedback": self.learning_metrics.total_feedback,
                "positive_feedback": self.learning_metrics.positive_feedback,
                "negative_feedback": self.learning_metrics.negative_feedback,
                "accuracy": self.learning_metrics.positive_feedback / max(1, self.learning_metrics.total_feedback),
                "embedding_improvements": self.learning_metrics.embedding_improvements,
                "ranking_improvements": self.learning_metrics.ranking_improvements,
                "training_examples_added": self.learning_metrics.training_examples_added,
                "accuracy_trend": self.learning_metrics.accuracy_trend[-10:] if self.learning_metrics.accuracy_trend else [],
                "top_query_patterns": list(self.embedding_weights.keys())[:5]
            }
            
            return insights
            
        except Exception as e:
            print(f"✗ Failed to get learning insights: {e}")
            return {"error": str(e)}
    
    def process_from_multimodal_agent(self, 
                                    multimodal_data: Dict[str, Any],
                                    session_id: str,
                                    user_id: str = "system") -> Dict[str, Any]:
        """
        Process data from multimodal agent and prepare for orchestrator.
        This method takes the output from the multimodal agent and prepares it for orchestrator consumption.
        
        Args:
            multimodal_data: Data from multimodal agent (from prepare_for_feedback_agent)
            session_id: Session ID for tracking
            user_id: ID of the user
            
        Returns:
            Dictionary containing processed data for orchestrator
        """
        try:
            if "error" in multimodal_data:
                print(f"[process_from_multimodal_agent] Error in multimodal data: {multimodal_data['error']}")
                return {"error": multimodal_data['error'], "orchestrator_ready": False}
            
            # Process multimodal data for orchestrator
            orchestrator_data = {
                "session_id": session_id,
                "user_id": user_id,
                "multimodal_results_count": multimodal_data.get("multimodal_results_count", 0),
                "content_types": multimodal_data.get("content_types", []),
                "average_confidence": multimodal_data.get("average_confidence", 0.0),
                "results_summary": multimodal_data.get("results_summary", []),
                "explainability_context": multimodal_data.get("explainability_context", {}),
                "feedback_collected": False,  # Will be set to True when user provides feedback
                "learning_insights": self.get_learning_insights(),
                "orchestrator_ready": True,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            print(f"[process_from_multimodal_agent] Prepared orchestrator data for session {session_id}")
            return orchestrator_data
            
        except Exception as e:
            print(f"[process_from_multimodal_agent] Error: {e}")
            return {"error": str(e), "orchestrator_ready": False}
    
    def get_workflow_learning_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive learning summary for the orchestrator.
        This method provides insights about the learning progress across the entire workflow.
        
        Returns:
            Dictionary containing comprehensive learning summary
        """
        try:
            insights = self.get_learning_insights()
            
            # Get recent feedback patterns
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT feedback_type, COUNT(*) as count
                FROM feedback
                WHERE timestamp >= datetime('now', '-7 days')
                GROUP BY feedback_type
                ORDER BY count DESC
            """)
            
            recent_feedback_patterns = dict(cursor.fetchall())
            
            # Get learning metrics trends
            cursor.execute("""
                SELECT metric_name, metric_value, timestamp
                FROM learning_metrics
                WHERE timestamp >= datetime('now', '-7 days')
                ORDER BY timestamp DESC
                LIMIT 50
            """)
            
            learning_trends = cursor.fetchall()
            
            # Calculate system health score
            health_score = 0.0
            if insights.get("total_feedback", 0) > 0:
                accuracy = insights.get("accuracy", 0.0)
                health_score = accuracy * 100  # Convert to percentage
            
            workflow_summary = {
                "learning_insights": insights,
                "recent_feedback_patterns": recent_feedback_patterns,
                "learning_trends": learning_trends,
                "system_health_score": health_score,
                "recommendations": self._generate_learning_recommendations(insights),
                "orchestrator_summary": True,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"[get_workflow_learning_summary] Generated comprehensive learning summary")
            return workflow_summary
            
        except Exception as e:
            print(f"[get_workflow_learning_summary] Error: {e}")
            return {"error": str(e), "orchestrator_summary": False}
    
    def _generate_learning_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate learning recommendations based on insights"""
        recommendations = []
        
        try:
            total_feedback = insights.get("total_feedback", 0)
            accuracy = insights.get("accuracy", 0.0)
            
            if total_feedback < 10:
                recommendations.append("Collect more user feedback to improve learning accuracy")
            
            if accuracy < 0.7:
                recommendations.append("Consider adjusting embedding weights or ranking algorithms")
            
            if insights.get("embedding_improvements", 0) < 5:
                recommendations.append("Increase embedding weight updates for better search relevance")
            
            if insights.get("ranking_improvements", 0) < 3:
                recommendations.append("Focus on improving resolution ranking logic")
            
            if not recommendations:
                recommendations.append("System learning is performing well - continue monitoring")
            
            return recommendations
            
        except Exception as e:
            print(f"[_generate_learning_recommendations] Error: {e}")
            return ["Error generating recommendations"]
    
    def get_training_examples(self, query: str, limit: int = 5) -> Dict[str, List[Dict]]:
        """Get relevant training examples for a query"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT positive_examples, negative_examples, usage_count
                FROM training_examples 
                WHERE query LIKE ? OR query LIKE ?
                ORDER BY usage_count DESC, created_at DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query.split()[0]}%", limit))
            
            results = cursor.fetchall()
            
            positive_examples = []
            negative_examples = []
            
            for row in results:
                if row[0]:  # positive_examples
                    positive_examples.extend(json.loads(row[0]))
                if row[1]:  # negative_examples
                    negative_examples.extend(json.loads(row[1]))
            
            return {
                "positive_examples": positive_examples[:limit],
                "negative_examples": negative_examples[:limit]
            }
            
        except Exception as e:
            print(f"✗ Failed to get training examples: {e}")
            return {"positive_examples": [], "negative_examples": []}


def ensure_feedback_agent_ready(agent: 'FeedbackIntegrationAgent') -> bool:
    """
    Ensure the feedback integration agent is ready for the workflow.

    Args:
        agent: FeedbackIntegrationAgent instance

    Returns:
        True if feedback agent is ready, False otherwise
    """
    try:
        # Test the agent by getting learning insights
        insights = agent.get_learning_insights()

        if insights and not insights.get("error"):
            print(f"[ensure_feedback_agent_ready] Feedback integration agent ready")
            return True
        else:
            print(f"[ensure_feedback_agent_ready] Feedback integration agent not responding properly")
            return False

    except Exception as e:
        print(f"[ensure_feedback_agent_ready] Error: {e}")
        return False
def main():
    """
    Main function demonstrating the FeedbackIntegrationAgent usage
    """
    try:
        # Initialize the feedback agent
        print("🚀 Initializing Feedback Integration Agent...")
        feedback_agent = FeedbackIntegrationAgent()
        
        # Sample search results and recommendations (would come from other agents)
        sample_search_results = [
            SearchResult(
                log_id="LOG-001",
                content="Database connection timeout issues",
                similarity_score=0.95,
                metadata={"service": "payment-service", "priority": "High"},
                related_context={}
            )
        ]
        
        sample_recommendations = [
            ResolutionRecommendation(
                title="Increase Connection Pool Size",
                description="Scale up database connection pool to handle load",
                priority="High",
                confidence_score=0.9,
                estimated_time="15 minutes",
                steps=["Check current pool size", "Increase pool size", "Monitor performance"],
                prerequisites=["Database access", "Monitoring tools"],
                related_artifacts=["LOG-001"]
            )
        ]
        
        # Simulate different types of feedback
        print("\n📝 Collecting user feedback...")
        
        # Thumbs up feedback
        feedback_id1 = feedback_agent.collect_feedback(
            user_id="user123",
            session_id="session456",
            query="database timeout issues",
            search_results=sample_search_results,
            recommendations=sample_recommendations,
            action_plan=None,
            feedback_type="thumbs_up",
            feedback_value=True
        )
        
        # Natural language feedback
        feedback_id2 = feedback_agent.collect_feedback(
            user_id="user456",
            session_id="session789",
            query="database timeout issues",
            search_results=sample_search_results,
            recommendations=sample_recommendations,
            action_plan=None,
            feedback_type="natural_language",
            feedback_value="This was helpful but the steps were too generic",
            feedback_text="This was helpful but the steps were too generic. Need more specific commands."
        )
        
        # Rating feedback
        feedback_id3 = feedback_agent.collect_feedback(
            user_id="user789",
            session_id="session101",
            query="database timeout issues",
            search_results=sample_search_results,
            recommendations=sample_recommendations,
            action_plan=None,
            feedback_type="rating",
            feedback_value=4  # 4 out of 5
        )
        
        # Get learning insights
        print("\n📊 Learning Insights:")
        insights = feedback_agent.get_learning_insights()
        for key, value in insights.items():
            print(f"  {key}: {value}")
        
        # Get training examples
        print("\n🎯 Training Examples:")
        examples = feedback_agent.get_training_examples("database timeout")
        print(f"  Positive examples: {len(examples['positive_examples'])}")
        print(f"  Negative examples: {len(examples['negative_examples'])}")
        
        print("\n✅ Feedback Integration Agent demonstration completed successfully!")
        print("💡 The system learns from every interaction!")
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
