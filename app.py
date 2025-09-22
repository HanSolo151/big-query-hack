"""Flask Backend API for DevOps Intelligence Platform
Unified with OrchestratorAgent for full pipeline execution """

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import asyncio
from datetime import datetime

# Import orchestrator
from ORCHESTRATOR_AGENT import OrchestratorAgent
from google.cloud import bigquery   # <-- added

app = Flask(__name__)
CORS(app)

# Global orchestrator instance
orchestrator = None

from datetime import datetime

def safe_parse_date(value):
    """Convert string to datetime if possible, else None"""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None

def initialize_vector_store(orchestrator):
    """Pull incidents from BigQuery and seed the vector store (non-destructive)"""
    try:
        print("⚡ Initializing vector store from BigQuery logs...")

        client = bigquery.Client()
        query = """
        SELECT
            log_id,
            CONCAT(
                IFNULL(title, ''), ' ',
                IFNULL(description, ''), ' ',
                IFNULL(resolution, ''), ' ',
                IFNULL(docs_faq, '')
            ) AS content,
            service,
            environment,
            cluster,
            namespace,
            pod,
            container,
            status,
            priority,
            severity,
            category,
            source_type,
            tool,
            commit_sha,
            file_path,
            incident_id,
            STRING(created_at) AS created_at,
            STRING(updated_at) AS updated_at
        FROM `big-station-472112-i1.log_dataset.log_vectors`
        LIMIT 500
        """
        rows = client.query(query).result()

        dataset = []
        for row in rows:
            dataset.append({
                "log_id": row.log_id,
                "content": row.content,
                "artifact_text":row.content,
                "metadata": {
                    "service": str(row.service),
                    "environment": str(row.environment),
                    "cluster": str(row.cluster),
                    "namespace": str(row.namespace),
                    "pod": str(row.pod),
                    "container": str(row.container),
                    "status": str(row.status),
                    "priority": str(row.priority),
                    "severity": str(row.severity),
                    "category": str(row.category),
                    "source_type": str(row.source_type),
                    "tool": str(row.tool),
                    "commit_sha": str(row.commit_sha),
                    "file_path": str(row.file_path),
                    "incident_id": str(row.incident_id),
                    #"tags": str(row.tags),
                    "created_at": safe_parse_date(row.created_at),
                    "updated_at": safe_parse_date(row.updated_at),
                }
            })

        if not dataset:
            print("⚠ No logs found in BigQuery. Vector store not seeded.")
            return False

        # Force recreate vector store fresh each run
        orchestrator.search_agent.create_vector_store(force_recreate=False)
        orchestrator.search_agent.add_logs_to_vector_store(dataset)
        orchestrator.search_agent._artifact_to_text = lambda artifact: artifact.get("content", "")
        orchestrator.search_agent._artifact_to_metadata = _artifact_to_metadata

        orchestrator.search_agent.add_logs_to_vector_store(dataset)
        print(f"✓ Loaded {len(dataset)} logs from BigQuery into vector store")
        return True

    except Exception as e:
        print(f"✗ Failed to initialize vector store from BigQuery: {e}")
        return False


def initialize_orchestrator():
    """Initialize OrchestratorAgent with all agents"""
    global orchestrator
    try:
        print("🚀 Initializing OrchestratorAgent...")
        orchestrator = OrchestratorAgent()
        orchestrator.start_proactive_monitoring()

        # Seed the vector store with BigQuery data
        initialize_vector_store(orchestrator)

        print("✓ OrchestratorAgent ready")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize orchestrator: {e}")
        return False


@app.route('/')
def index():
    """Serve a simple landing page"""
    return jsonify({"message": "Flask backend running. Visit Streamlit UI at http://localhost:8502"})


@app.route('/api/health')
def health():
    """Health check endpoint"""
    if not orchestrator:
        return jsonify({'status': 'not ready'}), 500

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'agents': {
            'embedding': orchestrator.embedding_agent is not None,
            'search': orchestrator.search_agent is not None,
            'resolution': orchestrator.resolution_agent is not None,
            'explainability': orchestrator.explainability_agent is not None,
            'multimodal': orchestrator.multimodal_agent is not None,
            'feedback': orchestrator.feedback_agent is not None,
            'proactive': orchestrator.proactive_agent_running,
        }
    })


@app.route('/api/search', methods=['POST'])
def search():
    """Run a query through orchestrator (search only)"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        k = data.get('k', 5)

        if not orchestrator:
            return jsonify({'error': 'Orchestrator not initialized'}), 500

        result = asyncio.run(orchestrator.process_incident_query(query, user_id="api", k_search=k))
        return jsonify({
            'query': query,
            'results': [sr.__dict__ for sr in result.search_results],
            'total_results': len(result.search_results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/resolution', methods=['POST'])
def resolution():
    """Run a query through orchestrator (resolution pipeline)"""
    try:
        data = request.get_json()
        query = data.get('query', '')

        if not orchestrator:
            return jsonify({'error': 'Orchestrator not initialized'}), 500

        result = asyncio.run(orchestrator.process_incident_query(query, user_id="api"))

        return jsonify({
            'query': query,
            'recommendations': [rec.__dict__ for rec in result.resolution_recommendations],
            'action_plan': result.action_plan.__dict__ if result.action_plan else None,
            'search_results_count': len(result.search_results),
            'explainability': result.explainability_result.__dict__ if result.explainability_result else None,
            'multimodal_results': [mm.__dict__ for mm in result.multimodal_results],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Submit feedback to feedback agent"""
    try:
        data = request.get_json()
        if not orchestrator or not orchestrator.feedback_agent:
            return jsonify({'error': 'Feedback agent not initialized'}), 500

        feedback_id = orchestrator.feedback_agent.collect_feedback(
            user_id=data.get('user_id', 'anonymous'),
            session_id=data.get('session_id', f'session_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
            query=data.get('query', ''),
            search_results=[],
            recommendations=[],
            action_plan=None,
            feedback_type=data.get('feedback_type', 'thumbs_up'),
            feedback_value=data.get('feedback_value', True),
            feedback_text=data.get('feedback_text')
        )

        return jsonify({
            'feedback_id': feedback_id,
            'message': 'Feedback submitted successfully',
            'learning_insights': orchestrator.feedback_agent.learning_metrics.__dict__
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/proactive/alerts')
def proactive_alerts():
    """Expose proactive agent alerts"""
    try:
        if not orchestrator or not orchestrator.proactive_agent:
            return jsonify({'error': 'Proactive agent not initialized'}), 500

        alerts = orchestrator.proactive_agent.get_active_alerts()
        issues = orchestrator.proactive_agent.get_emerging_issues()

        return jsonify({
            'alerts': [a.__dict__ for a in alerts],
            'issues': [i.__dict__ for i in issues],
            'total_alerts': len(alerts),
            'total_issues': len(issues)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    if initialize_orchestrator():
        print("✓ Backend ready. Starting Flask...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("✗ Failed to initialize backend")
