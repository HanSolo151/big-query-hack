"""
Flask Backend API for DevOps Intelligence Platform
Provides REST API endpoints for interacting with all agents
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
from datetime import datetime
import threading
import time

# Import agents
from SEARCH_AGENT_1 import VectorSearchAgent, LogData, ensure_search_agent_ready
from RESOLUTION_AGENT_ import ResolutionAgent
from FEEDBACK_INTEGRATION_AGENT import FeedbackIntegrationAgent
from PROACTIVE_AGENT import ProactiveAgent

app = Flask(__name__)
CORS(app)

# Global agent instances
search_agent = None
resolution_agent = None
feedback_agent = None
proactive_agent = None

def initialize_agents():
    """Initialize all agents"""
    global search_agent, resolution_agent, feedback_agent, proactive_agent
    
    try:
        print("🚀 Initializing agents...")
        
        # Initialize Search Agent
        search_agent = VectorSearchAgent()
        if ensure_search_agent_ready(search_agent, force_recreate=False):
            print("✓ Search Agent ready")
        else:
            print("✗ Search Agent initialization failed")
            return False
        
        # Initialize Resolution Agent
        resolution_agent = ResolutionAgent()
        print("✓ Resolution Agent ready")
        
        # Initialize Feedback Agent
        feedback_agent = FeedbackIntegrationAgent()
        print("✓ Feedback Agent ready")
        
        # Initialize Proactive Agent
        proactive_agent = ProactiveAgent()
        print("✓ Proactive Agent ready")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        return False

@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'agents': {
            'search': search_agent is not None,
            'resolution': resolution_agent is not None,
            'feedback': feedback_agent is not None,
            'proactive': proactive_agent is not None
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/search', methods=['POST'])
def search():
    """Search endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        k = data.get('k', 5)
        filter_metadata = data.get('filter_metadata', None)
        
        if not search_agent:
            return jsonify({'error': 'Search agent not initialized'}), 500
        
        # Perform search
        results = search_agent.vector_search(query, k=k, filter_metadata=filter_metadata)
        
        # Convert results to JSON-serializable format
        search_results = []
        for result in results:
            search_results.append({
                'log_id': result.log_id,
                'content': result.content,
                'similarity_score': result.similarity_score,
                'metadata': result.metadata,
                'related_context': result.related_context
            })
        
        return jsonify({
            'query': query,
            'results': search_results,
            'total_results': len(search_results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/resolution', methods=['POST'])
def get_resolution():
    """Get resolution recommendations"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        k = data.get('k', 5)
        
        if not search_agent or not resolution_agent:
            return jsonify({'error': 'Agents not initialized'}), 500
        
        # Search for similar incidents
        search_results = search_agent.search_for_resolution_agent(query, k=k)
        
        # Generate recommendations
        recommendations = resolution_agent.summarize_solutions(search_results, query)
        
        # Create action plan if complex enough
        action_plan = resolution_agent.create_action_plan(search_results, query)
        
        # Convert to JSON-serializable format
        recommendations_data = []
        for rec in recommendations:
            recommendations_data.append({
                'title': rec.title,
                'description': rec.description,
                'priority': rec.priority,
                'confidence_score': rec.confidence_score,
                'estimated_time': rec.estimated_time,
                'steps': rec.steps,
                'prerequisites': rec.prerequisites,
                'related_artifacts': rec.related_artifacts
            })
        
        action_plan_data = None
        if action_plan:
            action_plan_data = {
                'plan_id': action_plan.plan_id,
                'title': action_plan.title,
                'description': action_plan.description,
                'total_estimated_time': action_plan.total_estimated_time,
                'steps': action_plan.steps,
                'success_criteria': action_plan.success_criteria,
                'rollback_plan': action_plan.rollback_plan
            }
        
        return jsonify({
            'query': query,
            'recommendations': recommendations_data,
            'action_plan': action_plan_data,
            'search_results_count': len(search_results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback"""
    try:
        data = request.get_json()
        
        if not feedback_agent:
            return jsonify({'error': 'Feedback agent not initialized'}), 500
        
        # Extract feedback data
        user_id = data.get('user_id', 'anonymous')
        session_id = data.get('session_id', f'session_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        query = data.get('query', '')
        feedback_type = data.get('feedback_type', 'thumbs_up')
        feedback_value = data.get('feedback_value', True)
        feedback_text = data.get('feedback_text', None)
        
        # Mock search results and recommendations for feedback
        search_results = []
        recommendations = []
        action_plan = None
        
        if query and search_agent and resolution_agent:
            search_results = search_agent.vector_search(query, k=3)
            recommendations = resolution_agent.summarize_solutions(search_results, query)
            action_plan = resolution_agent.create_action_plan(search_results, query)
        
        # Submit feedback
        feedback_id = feedback_agent.collect_feedback(
            user_id=user_id,
            session_id=session_id,
            query=query,
            search_results=search_results,
            recommendations=recommendations,
            action_plan=action_plan,
            feedback_type=feedback_type,
            feedback_value=feedback_value,
            feedback_text=feedback_text
        )
        
        return jsonify({
            'feedback_id': feedback_id,
            'message': 'Feedback submitted successfully',
            'learning_insights': feedback_agent.get_learning_insights()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/proactive/alerts')
def get_proactive_alerts():
    """Get proactive alerts"""
    try:
        if not proactive_agent:
            return jsonify({'error': 'Proactive agent not initialized'}), 500
        
        alerts = proactive_agent.get_active_alerts()
        issues = proactive_agent.get_emerging_issues()
        clusters = proactive_agent.get_active_clusters()
        
        # Convert to JSON-serializable format
        alerts_data = []
        for alert in alerts:
            alerts_data.append({
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type,
                'title': alert.title,
                'message': alert.message,
                'severity': alert.severity,
                'confidence': alert.confidence,
                'affected_entities': alert.affected_entities,
                'recommended_actions': alert.recommended_actions,
                'escalation_timeframe': alert.escalation_timeframe,
                'created_at': alert.created_at.isoformat(),
                'acknowledged': alert.acknowledged,
                'resolved': alert.resolved
            })
        
        issues_data = []
        for issue in issues:
            issues_data.append({
                'issue_id': issue.issue_id,
                'title': issue.title,
                'description': issue.description,
                'severity': issue.severity,
                'confidence_score': issue.confidence_score,
                'affected_services': issue.affected_services,
                'affected_environments': issue.affected_environments,
                'ticket_count': issue.ticket_count,
                'growth_rate': issue.growth_rate,
                'predicted_escalation_time': issue.predicted_escalation_time.isoformat() if issue.predicted_escalation_time else None,
                'root_cause_hypothesis': issue.root_cause_hypothesis,
                'recommended_actions': issue.recommended_actions,
                'first_detected': issue.first_detected.isoformat()
            })
        
        clusters_data = []
        for cluster in clusters:
            clusters_data.append({
                'cluster_id': cluster.cluster_id,
                'ticket_ids': cluster.ticket_ids,
                'common_patterns': cluster.common_patterns,
                'severity_distribution': cluster.severity_distribution,
                'service_distribution': cluster.service_distribution,
                'environment_distribution': cluster.environment_distribution,
                'cluster_size': cluster.cluster_size,
                'avg_similarity': cluster.avg_similarity,
                'created_at': cluster.created_at.isoformat()
            })
        
        return jsonify({
            'alerts': alerts_data,
            'emerging_issues': issues_data,
            'clusters': clusters_data,
            'total_alerts': len(alerts),
            'total_issues': len(issues),
            'total_clusters': len(clusters)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/proactive/start', methods=['POST'])
def start_proactive_monitoring():
    """Start proactive monitoring"""
    try:
        if not proactive_agent:
            return jsonify({'error': 'Proactive agent not initialized'}), 500
        
        proactive_agent.start_monitoring()
        
        return jsonify({
            'message': 'Proactive monitoring started',
            'status': 'active'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/proactive/stop', methods=['POST'])
def stop_proactive_monitoring():
    """Stop proactive monitoring"""
    try:
        if not proactive_agent:
            return jsonify({'error': 'Proactive agent not initialized'}), 500
        
        proactive_agent.stop_monitoring()
        
        return jsonify({
            'message': 'Proactive monitoring stopped',
            'status': 'inactive'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/proactive/acknowledge/<alert_id>', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    try:
        if not proactive_agent:
            return jsonify({'error': 'Proactive agent not initialized'}), 500
        
        success = proactive_agent.acknowledge_alert(alert_id)
        
        if success:
            return jsonify({'message': f'Alert {alert_id} acknowledged'})
        else:
            return jsonify({'error': 'Alert not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/load', methods=['POST'])
def load_incident_data():
    """Load incident data from JSON file"""
    try:
        if not search_agent:
            return jsonify({'error': 'Search agent not initialized'}), 500
        
        # Load incident data from JSON file
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
        
        # Add logs to vector store
        search_agent.add_logs_to_vector_store(sample_logs)
        
        return jsonify({
            'message': f'Loaded {len(sample_logs)} incident records',
            'records_loaded': len(sample_logs)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize agents
    if initialize_agents():
        print("🚀 All agents initialized successfully!")
        print("🌐 Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize agents. Please check your configuration.")
