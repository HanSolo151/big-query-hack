"""
Proactive Agent - Proactive Intelligence
Continuously scans newly incoming tickets, performs clustering to identify patterns,
and provides early warning alerts for emerging issues before escalation.
Transforms the system from reactive to proactive intelligence.
"""

import os
import json
import time
import threading
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage

# Import from other agents
from SEARCH_AGENT_1 import SearchResult, LogData, VectorSearchAgent
from RESOLUTION_AGENT_ import ResolutionRecommendation, ResolutionAgent
from FEEDBACK_INTEGRATION_AGENT import FeedbackIntegrationAgent


@dataclass
class TicketCluster:
    """Data class for ticket clusters"""
    cluster_id: str
    center_embedding: List[float]
    ticket_ids: List[str]
    common_patterns: List[str]
    severity_distribution: Dict[str, int]
    service_distribution: Dict[str, int]
    environment_distribution: Dict[str, int]
    time_range: Tuple[datetime, datetime]
    cluster_size: int
    avg_similarity: float
    created_at: datetime


@dataclass
class EmergingIssue:
    """Data class for emerging issues detected proactively"""
    issue_id: str
    title: str
    description: str
    severity: str  # Low, Medium, High, Critical
    confidence_score: float  # 0.0 to 1.0
    affected_services: List[str]
    affected_environments: List[str]
    ticket_count: int
    growth_rate: float  # Tickets per hour
    predicted_escalation_time: Optional[datetime]
    root_cause_hypothesis: List[str]
    recommended_actions: List[str]
    clusters_involved: List[str]
    first_detected: datetime
    last_updated: datetime


@dataclass
class ProactiveAlert:
    """Data class for proactive alerts"""
    alert_id: str
    alert_type: str  # "emerging_issue", "cluster_anomaly", "escalation_risk"
    title: str
    message: str
    severity: str
    confidence: float
    affected_entities: List[str]
    recommended_actions: List[str]
    escalation_timeframe: Optional[str]
    created_at: datetime
    acknowledged: bool = False
    resolved: bool = False


class ProactiveAgent:
    """
    Proactive Agent that continuously monitors tickets and provides early warnings
    """
    
    def __init__(self, 
                 project_id: str = "big-station-472112-i1",
                 credentials_path: str = "big-station-472112-i1-01b16573569e.json",
                 api_key_path: str = "Gemini_API_Key.txt",
                 db_path: str = "proactive_monitoring.db"):
        """
        Initialize the ProactiveAgent
        
        Args:
            project_id: Google Cloud project ID
            credentials_path: Path to service account credentials
            api_key_path: Path to Gemini API key
            db_path: Path to SQLite database for monitoring data
        """
        self.project_id = project_id
        self.credentials_path = credentials_path
        self.api_key_path = api_key_path
        self.db_path = db_path
        
        # Initialize components
        self._setup_database()
        self._setup_gemini()
        self._setup_embeddings()
        self._setup_search_agent()
        
        # Monitoring parameters
        self.scan_interval = 300  # 5 minutes
        self.cluster_min_samples = 3
        self.cluster_eps = 0.3
        self.alert_thresholds = {
            "cluster_size": 5,
            "growth_rate": 2.0,  # tickets per hour
            "similarity_threshold": 0.7
        }
        
        # State tracking
        self.active_clusters: Dict[str, TicketCluster] = {}
        self.emerging_issues: Dict[str, EmergingIssue] = {}
        self.active_alerts: Dict[str, ProactiveAlert] = {}
        self.last_scan_time = datetime.now()
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        print("✓ Proactive Agent initialized successfully")
    
    def _setup_database(self):
        """Set up SQLite database for proactive monitoring"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # Create tickets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    source_type TEXT,
                    service TEXT,
                    environment TEXT,
                    severity TEXT,
                    status TEXT,
                    created_at TEXT,
                    embedding_vector TEXT,
                    processed BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Create clusters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    cluster_id TEXT PRIMARY KEY,
                    center_embedding TEXT,
                    ticket_ids TEXT,
                    common_patterns TEXT,
                    severity_distribution TEXT,
                    service_distribution TEXT,
                    environment_distribution TEXT,
                    time_range TEXT,
                    cluster_size INTEGER,
                    avg_similarity REAL,
                    created_at TEXT,
                    last_updated TEXT
                )
            """)
            
            # Create emerging issues table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emerging_issues (
                    issue_id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    severity TEXT,
                    confidence_score REAL,
                    affected_services TEXT,
                    affected_environments TEXT,
                    ticket_count INTEGER,
                    growth_rate REAL,
                    predicted_escalation_time TEXT,
                    root_cause_hypothesis TEXT,
                    recommended_actions TEXT,
                    clusters_involved TEXT,
                    first_detected TEXT,
                    last_updated TEXT
                )
            """)
            
            # Create alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    alert_type TEXT,
                    title TEXT,
                    message TEXT,
                    severity TEXT,
                    confidence REAL,
                    affected_entities TEXT,
                    recommended_actions TEXT,
                    escalation_timeframe TEXT,
                    created_at TEXT,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            self.conn.commit()
            print("✓ Proactive monitoring database setup completed")
            
        except Exception as e:
            print(f"✗ Database setup failed: {e}")
            raise
    
    def _setup_gemini(self):
        """Set up Gemini for proactive analysis"""
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
    
    def _setup_embeddings(self):
        """Set up embeddings for clustering"""
        try:
            with open(self.api_key_path, 'r') as f:
                api_key = f.read().strip()
            
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
            
            print("✓ Embeddings setup completed")
            
        except Exception as e:
            print(f"✗ Embeddings setup failed: {e}")
            raise
    
    def _setup_search_agent(self):
        """Set up search agent for ticket retrieval"""
        try:
            self.search_agent = VectorSearchAgent(
                project_id=self.project_id,
                credentials_path=self.credentials_path,
                api_key_path=self.api_key_path
            )
            print("✓ Search agent setup completed")
            
        except Exception as e:
            print(f"✗ Search agent setup failed: {e}")
            raise
    
    def start_monitoring(self):
        """Start continuous monitoring of incoming tickets"""
        try:
            if self.monitoring_active:
                print("! Monitoring already active")
                return
            
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            print("✓ Proactive monitoring started")
            
        except Exception as e:
            print(f"✗ Failed to start monitoring: {e}")
            raise
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        try:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            
            print("✓ Proactive monitoring stopped")
            
        except Exception as e:
            print(f"✗ Failed to stop monitoring: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Scan for new tickets
                new_tickets = self._scan_new_tickets()
                
                if new_tickets:
                    print(f"📊 Found {len(new_tickets)} new tickets")
                    
                    # Process new tickets
                    self._process_new_tickets(new_tickets)
                    
                    # Update clusters
                    self._update_clusters()
                    
                    # Detect emerging issues
                    self._detect_emerging_issues()
                    
                    # Generate alerts
                    self._generate_proactive_alerts()
                
                # Wait for next scan
                time.sleep(self.scan_interval)
                
            except Exception as e:
                print(f"✗ Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _scan_new_tickets(self) -> List[LogData]:
        """Scan for newly incoming tickets"""
        try:
            # In a real implementation, this would query your ticket system
            # For demo, we'll simulate by checking for unprocessed tickets in our dataset
            
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM tickets 
                WHERE processed = FALSE 
                ORDER BY created_at DESC 
                LIMIT 50
            """)
            
            rows = cursor.fetchall()
            new_tickets = []
            
            for row in rows:
                ticket_data = LogData(
                    log_id=row[0],
                    title=row[1],
                    description=row[2],
                    source_type=row[3],
                    service=row[4],
                    environment=row[5],
                    severity=row[6],
                    status=row[7],
                    created_at=datetime.fromisoformat(row[8]) if row[8] else datetime.now()
                )
                new_tickets.append(ticket_data)
            
            return new_tickets
            
        except Exception as e:
            print(f"✗ Failed to scan new tickets: {e}")
            return []
    
    def _process_new_tickets(self, tickets: List[LogData]):
        """Process new tickets and generate embeddings"""
        try:
            cursor = self.conn.cursor()
            
            for ticket in tickets:
                # Generate embedding
                ticket_text = f"{ticket.title} {ticket.description}"
                embedding = self.embeddings.embed_query(ticket_text)
                
                # Store in database
                cursor.execute("""
                    INSERT OR REPLACE INTO tickets 
                    (ticket_id, title, description, source_type, service, environment, 
                     severity, status, created_at, embedding_vector, processed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticket.log_id,
                    ticket.title,
                    ticket.description,
                    ticket.source_type,
                    ticket.service,
                    ticket.environment,
                    ticket.severity,
                    ticket.status,
                    ticket.created_at.isoformat(),
                    json.dumps(embedding),
                    True
                ))
            
            self.conn.commit()
            print(f"✓ Processed {len(tickets)} new tickets")
            
        except Exception as e:
            print(f"✗ Failed to process new tickets: {e}")
    
    def _update_clusters(self):
        """Update ticket clusters using DBSCAN"""
        try:
            # Get recent tickets for clustering
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT ticket_id, embedding_vector, service, environment, severity, created_at
                FROM tickets 
                WHERE created_at >= datetime('now', '-24 hours')
                AND status != 'resolved'
                ORDER BY created_at DESC
            """)
            
            rows = cursor.fetchall()
            
            if len(rows) < self.cluster_min_samples:
                return
            
            # Prepare embeddings and metadata
            embeddings = []
            ticket_metadata = []
            
            for row in rows:
                embedding = json.loads(row[1])
                embeddings.append(embedding)
                ticket_metadata.append({
                    'ticket_id': row[0],
                    'service': row[2],
                    'environment': row[3],
                    'severity': row[4],
                    'created_at': row[5]
                })
            
            # Perform clustering
            clustering = DBSCAN(
                eps=self.cluster_eps,
                min_samples=self.cluster_min_samples,
                metric='cosine'
            ).fit(embeddings)
            
            # Process clusters
            self._process_clusters(clustering, ticket_metadata, embeddings)
            
        except Exception as e:
            print(f"✗ Failed to update clusters: {e}")
    
    def _process_clusters(self, clustering, ticket_metadata, embeddings):
        """Process clustering results and update cluster database"""
        try:
            unique_labels = set(clustering.labels_)
            unique_labels.discard(-1)  # Remove noise label
            
            cursor = self.conn.cursor()
            
            for cluster_label in unique_labels:
                # Get tickets in this cluster
                cluster_indices = [i for i, label in enumerate(clustering.labels_) if label == cluster_label]
                cluster_tickets = [ticket_metadata[i] for i in cluster_indices]
                cluster_embeddings = [embeddings[i] for i in cluster_indices]
                
                # Calculate cluster center
                center_embedding = np.mean(cluster_embeddings, axis=0).tolist()
                
                # Analyze cluster patterns
                patterns = self._analyze_cluster_patterns(cluster_tickets)
                
                # Create cluster ID
                cluster_id = f"CLUSTER-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{cluster_label}"
                
                # Calculate similarity
                similarities = []
                for embedding in cluster_embeddings:
                    sim = cosine_similarity([embedding], [center_embedding])[0][0]
                    similarities.append(sim)
                avg_similarity = np.mean(similarities)
                
                # Create cluster object
                cluster = TicketCluster(
                    cluster_id=cluster_id,
                    center_embedding=center_embedding,
                    ticket_ids=[t['ticket_id'] for t in cluster_tickets],
                    common_patterns=patterns['common_patterns'],
                    severity_distribution=patterns['severity_distribution'],
                    service_distribution=patterns['service_distribution'],
                    environment_distribution=patterns['environment_distribution'],
                    time_range=patterns['time_range'],
                    cluster_size=len(cluster_tickets),
                    avg_similarity=avg_similarity,
                    created_at=datetime.now()
                )
                
                # Store cluster
                self._store_cluster(cluster)
                self.active_clusters[cluster_id] = cluster
                
                print(f"✓ Created cluster {cluster_id} with {len(cluster_tickets)} tickets")
            
        except Exception as e:
            print(f"✗ Failed to process clusters: {e}")
    
    def _analyze_cluster_patterns(self, cluster_tickets) -> Dict[str, Any]:
        """Analyze patterns within a cluster"""
        try:
            # Count distributions
            severity_dist = Counter(t['severity'] for t in cluster_tickets)
            service_dist = Counter(t['service'] for t in cluster_tickets)
            environment_dist = Counter(t['environment'] for t in cluster_tickets)
            
            # Time range
            timestamps = [datetime.fromisoformat(t['created_at']) for t in cluster_tickets]
            time_range = (min(timestamps), max(timestamps))
            
            # Common patterns (simplified)
            common_patterns = []
            if len(set(t['service'] for t in cluster_tickets)) == 1:
                common_patterns.append(f"Single service: {cluster_tickets[0]['service']}")
            if len(set(t['environment'] for t in cluster_tickets)) == 1:
                common_patterns.append(f"Single environment: {cluster_tickets[0]['environment']}")
            
            return {
                'common_patterns': common_patterns,
                'severity_distribution': dict(severity_dist),
                'service_distribution': dict(service_dist),
                'environment_distribution': dict(environment_dist),
                'time_range': time_range
            }
            
        except Exception as e:
            print(f"✗ Failed to analyze cluster patterns: {e}")
            return {
                'common_patterns': [],
                'severity_distribution': {},
                'service_distribution': {},
                'environment_distribution': {},
                'time_range': (datetime.now(), datetime.now())
            }
    
    def _store_cluster(self, cluster: TicketCluster):
        """Store cluster in database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO clusters 
                (cluster_id, center_embedding, ticket_ids, common_patterns, 
                 severity_distribution, service_distribution, environment_distribution,
                 time_range, cluster_size, avg_similarity, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cluster.cluster_id,
                json.dumps(cluster.center_embedding),
                json.dumps(cluster.ticket_ids),
                json.dumps(cluster.common_patterns),
                json.dumps(cluster.severity_distribution),
                json.dumps(cluster.service_distribution),
                json.dumps(cluster.environment_distribution),
                json.dumps([cluster.time_range[0].isoformat(), cluster.time_range[1].isoformat()]),
                cluster.cluster_size,
                cluster.avg_similarity,
                cluster.created_at.isoformat(),
                datetime.now().isoformat()
            ))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"✗ Failed to store cluster: {e}")
    
    def _detect_emerging_issues(self):
        """Detect emerging issues from clusters"""
        try:
            for cluster_id, cluster in self.active_clusters.items():
                # Check if cluster indicates an emerging issue
                if cluster.cluster_size >= self.alert_thresholds['cluster_size']:
                    # Calculate growth rate
                    growth_rate = self._calculate_growth_rate(cluster)
                    
                    if growth_rate >= self.alert_thresholds['growth_rate']:
                        # Generate emerging issue
                        issue = self._generate_emerging_issue(cluster, growth_rate)
                        if issue:
                            self.emerging_issues[issue.issue_id] = issue
                            self._store_emerging_issue(issue)
                            
                            print(f"🚨 Detected emerging issue: {issue.title}")
            
        except Exception as e:
            print(f"✗ Failed to detect emerging issues: {e}")
    
    def _calculate_growth_rate(self, cluster: TicketCluster) -> float:
        """Calculate ticket growth rate for a cluster"""
        try:
            # Get tickets from last hour
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM tickets 
                WHERE ticket_id IN ({})
                AND created_at >= datetime('now', '-1 hour')
            """.format(','.join(['?' for _ in cluster.ticket_ids])), cluster.ticket_ids)
            
            recent_count = cursor.fetchone()[0]
            return recent_count  # tickets per hour
            
        except Exception as e:
            print(f"✗ Failed to calculate growth rate: {e}")
            return 0.0
    
    def _generate_emerging_issue(self, cluster: TicketCluster, growth_rate: float) -> Optional[EmergingIssue]:
        """Generate emerging issue from cluster analysis"""
        try:
            # Use Gemini to analyze the cluster and generate issue description
            cluster_analysis = self._analyze_cluster_with_gemini(cluster)
            
            issue_id = f"EMERGING-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            issue = EmergingIssue(
                issue_id=issue_id,
                title=cluster_analysis.get('title', 'Emerging Issue Detected'),
                description=cluster_analysis.get('description', ''),
                severity=cluster_analysis.get('severity', 'Medium'),
                confidence_score=cluster_analysis.get('confidence', 0.7),
                affected_services=list(cluster.service_distribution.keys()),
                affected_environments=list(cluster.environment_distribution.keys()),
                ticket_count=cluster.cluster_size,
                growth_rate=growth_rate,
                predicted_escalation_time=cluster_analysis.get('escalation_time'),
                root_cause_hypothesis=cluster_analysis.get('root_causes', []),
                recommended_actions=cluster_analysis.get('recommendations', []),
                clusters_involved=[cluster.cluster_id],
                first_detected=datetime.now(),
                last_updated=datetime.now()
            )
            
            return issue
            
        except Exception as e:
            print(f"✗ Failed to generate emerging issue: {e}")
            return None
    
    def _analyze_cluster_with_gemini(self, cluster: TicketCluster) -> Dict[str, Any]:
        """Use Gemini to analyze cluster and generate insights"""
        try:
            prompt = f"""
            Analyze this ticket cluster and provide insights:
            
            Cluster Size: {cluster.cluster_size}
            Services: {cluster.service_distribution}
            Environments: {cluster.environment_distribution}
            Severity Distribution: {cluster.severity_distribution}
            Common Patterns: {cluster.common_patterns}
            Time Range: {cluster.time_range[0]} to {cluster.time_range[1]}
            
            Provide analysis in JSON format:
            {{
                "title": "Brief title for the emerging issue",
                "description": "Detailed description of the potential issue",
                "severity": "Low/Medium/High/Critical",
                "confidence": 0.0-1.0,
                "escalation_time": "Estimated time to escalation (ISO format or null)",
                "root_causes": ["hypothesis1", "hypothesis2"],
                "recommendations": ["action1", "action2"]
            }}
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            analysis = json.loads(response.content)
            
            return analysis
            
        except Exception as e:
            print(f"✗ Failed to analyze cluster with Gemini: {e}")
            return {
                "title": "Emerging Issue Detected",
                "description": "Multiple similar tickets detected",
                "severity": "Medium",
                "confidence": 0.5,
                "escalation_time": None,
                "root_causes": ["Unknown"],
                "recommendations": ["Investigate further"]
            }
    
    def _store_emerging_issue(self, issue: EmergingIssue):
        """Store emerging issue in database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO emerging_issues 
                (issue_id, title, description, severity, confidence_score,
                 affected_services, affected_environments, ticket_count, growth_rate,
                 predicted_escalation_time, root_cause_hypothesis, recommended_actions,
                 clusters_involved, first_detected, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                issue.issue_id,
                issue.title,
                issue.description,
                issue.severity,
                issue.confidence_score,
                json.dumps(issue.affected_services),
                json.dumps(issue.affected_environments),
                issue.ticket_count,
                issue.growth_rate,
                issue.predicted_escalation_time.isoformat() if issue.predicted_escalation_time else None,
                json.dumps(issue.root_cause_hypothesis),
                json.dumps(issue.recommended_actions),
                json.dumps(issue.clusters_involved),
                issue.first_detected.isoformat(),
                issue.last_updated.isoformat()
            ))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"✗ Failed to store emerging issue: {e}")
    
    def _generate_proactive_alerts(self):
        """Generate proactive alerts for emerging issues"""
        try:
            for issue_id, issue in self.emerging_issues.items():
                # Check if alert already exists
                if issue_id in self.active_alerts:
                    continue
                
                # Generate alert
                alert = self._create_proactive_alert(issue)
                if alert:
                    self.active_alerts[alert.alert_id] = alert
                    self._store_alert(alert)
                    
                    print(f"🚨 Generated alert: {alert.title}")
            
        except Exception as e:
            print(f"✗ Failed to generate proactive alerts: {e}")
    
    def _create_proactive_alert(self, issue: EmergingIssue) -> Optional[ProactiveAlert]:
        """Create proactive alert from emerging issue"""
        try:
            alert_id = f"ALERT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # Determine alert type and message
            if issue.growth_rate > 5:
                alert_type = "escalation_risk"
                message = f"High growth rate detected: {issue.growth_rate:.1f} tickets/hour. Immediate attention required."
            elif issue.ticket_count > 10:
                alert_type = "cluster_anomaly"
                message = f"Large cluster detected: {issue.ticket_count} similar tickets. Potential system-wide issue."
            else:
                alert_type = "emerging_issue"
                message = f"Emerging pattern detected: {issue.ticket_count} tickets with similar characteristics."
            
            alert = ProactiveAlert(
                alert_id=alert_id,
                alert_type=alert_type,
                title=f"Proactive Alert: {issue.title}",
                message=message,
                severity=issue.severity,
                confidence=issue.confidence_score,
                affected_entities=issue.affected_services + issue.affected_environments,
                recommended_actions=issue.recommended_actions,
                escalation_timeframe=f"{issue.predicted_escalation_time}" if issue.predicted_escalation_time else None,
                created_at=datetime.now()
            )
            
            return alert
            
        except Exception as e:
            print(f"✗ Failed to create proactive alert: {e}")
            return None
    
    def _store_alert(self, alert: ProactiveAlert):
        """Store alert in database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO alerts 
                (alert_id, alert_type, title, message, severity, confidence,
                 affected_entities, recommended_actions, escalation_timeframe,
                 created_at, acknowledged, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id,
                alert.alert_type,
                alert.title,
                alert.message,
                alert.severity,
                alert.confidence,
                json.dumps(alert.affected_entities),
                json.dumps(alert.recommended_actions),
                alert.escalation_timeframe,
                alert.created_at.isoformat(),
                alert.acknowledged,
                alert.resolved
            ))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"✗ Failed to store alert: {e}")
    
    def get_active_alerts(self) -> List[ProactiveAlert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_emerging_issues(self) -> List[EmergingIssue]:
        """Get all emerging issues"""
        return list(self.emerging_issues.values())
    
    def get_active_clusters(self) -> List[TicketCluster]:
        """Get all active clusters"""
        return list(self.active_clusters.values())
    
<<<<<<< HEAD
=======
    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status for the orchestrator.
        This method provides a complete status overview for the proactive agent.
        
        Returns:
            Dictionary containing comprehensive status information
        """
        try:
            status = {
                "monitoring_active": self.monitoring_active,
                "active_clusters_count": len(self.active_clusters),
                "emerging_issues_count": len(self.emerging_issues),
                "active_alerts_count": len(self.active_alerts),
                "last_scan_time": self.last_scan_time.isoformat(),
                "scan_interval": self.scan_interval,
                "alert_thresholds": self.alert_thresholds,
                "clusters": [
                    {
                        "cluster_id": cluster.cluster_id,
                        "cluster_size": cluster.cluster_size,
                        "avg_similarity": cluster.avg_similarity,
                        "common_patterns": cluster.common_patterns,
                        "severity_distribution": cluster.severity_distribution,
                        "service_distribution": cluster.service_distribution,
                        "created_at": cluster.created_at.isoformat()
                    }
                    for cluster in self.active_clusters.values()
                ],
                "emerging_issues": [
                    {
                        "issue_id": issue.issue_id,
                        "title": issue.title,
                        "severity": issue.severity,
                        "confidence_score": issue.confidence_score,
                        "ticket_count": issue.ticket_count,
                        "growth_rate": issue.growth_rate,
                        "affected_services": issue.affected_services,
                        "first_detected": issue.first_detected.isoformat()
                    }
                    for issue in self.emerging_issues.values()
                ],
                "alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "alert_type": alert.alert_type,
                        "title": alert.title,
                        "severity": alert.severity,
                        "confidence": alert.confidence,
                        "acknowledged": alert.acknowledged,
                        "resolved": alert.resolved,
                        "created_at": alert.created_at.isoformat()
                    }
                    for alert in self.active_alerts.values()
                ],
                "workflow_ready": True,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"[get_workflow_status] Generated comprehensive status for orchestrator")
            return status
            
        except Exception as e:
            print(f"[get_workflow_status] Error: {e}")
            return {"error": str(e), "workflow_ready": False}
    
    def integrate_with_orchestrator(self, orchestrator_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate proactive agent data with orchestrator.
        This method processes orchestrator data and provides proactive insights.
        
        Args:
            orchestrator_data: Data from orchestrator
            
        Returns:
            Dictionary containing integrated proactive insights
        """
        try:
            # Extract relevant information from orchestrator data
            session_id = orchestrator_data.get("session_id", "unknown")
            query = orchestrator_data.get("query", "unknown")
            
            # Check if current query matches any active clusters or emerging issues
            matching_clusters = []
            matching_issues = []
            matching_alerts = []
            
            for cluster in self.active_clusters.values():
                # Simple matching logic - could be enhanced with semantic similarity
                if any(keyword in query.lower() for keyword in cluster.common_patterns):
                    matching_clusters.append(cluster)
            
            for issue in self.emerging_issues.values():
                if any(keyword in query.lower() for keyword in issue.title.lower().split()):
                    matching_issues.append(issue)
            
            for alert in self.active_alerts.values():
                if any(keyword in query.lower() for keyword in alert.title.lower().split()):
                    matching_alerts.append(alert)
            
            # Generate proactive insights
            proactive_insights = {
                "session_id": session_id,
                "query": query,
                "matching_clusters": [
                    {
                        "cluster_id": cluster.cluster_id,
                        "cluster_size": cluster.cluster_size,
                        "common_patterns": cluster.common_patterns,
                        "severity_distribution": cluster.severity_distribution
                    }
                    for cluster in matching_clusters
                ],
                "matching_issues": [
                    {
                        "issue_id": issue.issue_id,
                        "title": issue.title,
                        "severity": issue.severity,
                        "confidence_score": issue.confidence_score,
                        "growth_rate": issue.growth_rate
                    }
                    for issue in matching_issues
                ],
                "matching_alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "title": alert.title,
                        "severity": alert.severity,
                        "confidence": alert.confidence
                    }
                    for alert in matching_alerts
                ],
                "proactive_recommendations": self._generate_proactive_recommendations(
                    matching_clusters, matching_issues, matching_alerts
                ),
                "orchestrator_integrated": True,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"[integrate_with_orchestrator] Generated proactive insights for session {session_id}")
            return proactive_insights
            
        except Exception as e:
            print(f"[integrate_with_orchestrator] Error: {e}")
            return {"error": str(e), "orchestrator_integrated": False}
    
    def _generate_proactive_recommendations(self, 
                                          clusters: List[TicketCluster],
                                          issues: List[EmergingIssue],
                                          alerts: List[ProactiveAlert]) -> List[str]:
        """Generate proactive recommendations based on current state"""
        recommendations = []
        
        try:
            if clusters:
                recommendations.append(f"Found {len(clusters)} related clusters - consider investigating patterns")
            
            if issues:
                high_severity_issues = [i for i in issues if i.severity in ["High", "Critical"]]
                if high_severity_issues:
                    recommendations.append(f"⚠️ {len(high_severity_issues)} high-severity emerging issues detected")
            
            if alerts:
                unacknowledged_alerts = [a for a in alerts if not a.acknowledged]
                if unacknowledged_alerts:
                    recommendations.append(f"🚨 {len(unacknowledged_alerts)} unacknowledged alerts require attention")
            
            if not recommendations:
                recommendations.append("No immediate proactive concerns detected")
            
            return recommendations
            
        except Exception as e:
            print(f"[_generate_proactive_recommendations] Error: {e}")
            return ["Error generating proactive recommendations"]
    
>>>>>>> 1191854 (agentic system)
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                
                # Update database
                cursor = self.conn.cursor()
                cursor.execute("""
                    UPDATE alerts SET acknowledged = TRUE WHERE alert_id = ?
                """, (alert_id,))
                self.conn.commit()
                
                return True
            return False
            
        except Exception as e:
            print(f"✗ Failed to acknowledge alert: {e}")
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                
                # Update database
                cursor = self.conn.cursor()
                cursor.execute("""
                    UPDATE alerts SET resolved = TRUE WHERE alert_id = ?
                """, (alert_id,))
                self.conn.commit()
                
                return True
            return False
            
        except Exception as e:
            print(f"✗ Failed to resolve alert: {e}")
            return False


<<<<<<< HEAD
=======
def ensure_proactive_agent_ready(agent: ProactiveAgent) -> bool:
    """Ensure the proactive agent is ready for the workflow.
    
    Args:
        agent: ProactiveAgent instance
        
    Returns:
        True if proactive agent is ready, False otherwise
    """
    try:
        # Test the agent by getting workflow status
        status = agent.get_workflow_status()
        
        if status and not status.get("error"):
            print(f"[ensure_proactive_agent_ready] Proactive agent ready")
            return True
        else:
            print(f"[ensure_proactive_agent_ready] Proactive agent not responding properly")
            return False
        
    except Exception as e:
        print(f"[ensure_proactive_agent_ready] Error: {e}")
        return False


>>>>>>> 1191854 (agentic system)
def main():
    """
    Main function demonstrating the ProactiveAgent usage
    """
    try:
        # Initialize the proactive agent
        print("🚀 Initializing Proactive Agent...")
        agent = ProactiveAgent()
        
        # Simulate some ticket data for demo
        print("\n📊 Simulating ticket data...")
        sample_tickets = [
            LogData(
                log_id="TICKET-001",
                title="Payment API timeout errors",
                description="Multiple payment API calls timing out",
                source_type="log",
                service="payment-service",
                environment="prod",
                severity="error",
                status="open",
                created_at=datetime.now() - timedelta(minutes=30)
            ),
            LogData(
                log_id="TICKET-002",
                title="Payment API timeout errors",
                description="Payment API calls timing out",
                source_type="log",
                service="payment-service",
                environment="prod",
                severity="error",
                status="open",
                created_at=datetime.now() - timedelta(minutes=25)
            ),
            LogData(
                log_id="TICKET-003",
                title="Payment API timeout errors",
                description="Payment service API timeout",
                source_type="log",
                service="payment-service",
                environment="prod",
                severity="error",
                status="open",
                created_at=datetime.now() - timedelta(minutes=20)
            ),
            LogData(
                log_id="TICKET-004",
                title="Payment API timeout errors",
                description="API timeout in payment service",
                source_type="log",
                service="payment-service",
                environment="prod",
                severity="error",
                status="open",
                created_at=datetime.now() - timedelta(minutes=15)
            ),
            LogData(
                log_id="TICKET-005",
                title="Payment API timeout errors",
                description="Payment API timeout issues",
                source_type="log",
                service="payment-service",
                environment="prod",
                severity="error",
                status="open",
                created_at=datetime.now() - timedelta(minutes=10)
            )
        ]
        
        # Process tickets
        agent._process_new_tickets(sample_tickets)
        
        # Update clusters
        agent._update_clusters()
        
        # Detect emerging issues
        agent._detect_emerging_issues()
        
        # Generate alerts
        agent._generate_proactive_alerts()
        
        # Display results
        print("\n🎯 Active Clusters:")
        clusters = agent.get_active_clusters()
        for cluster in clusters:
            print(f"  {cluster.cluster_id}: {cluster.cluster_size} tickets")
            print(f"    Services: {cluster.service_distribution}")
            print(f"    Severity: {cluster.severity_distribution}")
        
        print("\n🚨 Emerging Issues:")
        issues = agent.get_emerging_issues()
        for issue in issues:
            print(f"  {issue.title}")
            print(f"    Confidence: {issue.confidence_score:.2f}")
            print(f"    Growth Rate: {issue.growth_rate:.1f} tickets/hour")
            print(f"    Affected Services: {issue.affected_services}")
        
        print("\n🔔 Active Alerts:")
        alerts = agent.get_active_alerts()
        for alert in alerts:
            print(f"  {alert.title}")
            print(f"    Type: {alert.alert_type}")
            print(f"    Severity: {alert.severity}")
            print(f"    Message: {alert.message}")
        
        print("\n✅ Proactive Agent demonstration completed successfully!")
        print("💡 System transformed from reactive to proactive intelligence!")
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
