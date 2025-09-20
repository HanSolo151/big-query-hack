"""
Streamlit UI for DevOps Intelligence Platform
Interactive web interface for all agents
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd

# Configuration
API_BASE_URL = "http://localhost:5000/api"

# Page configuration
st.set_page_config(
    page_title="DevOps Intelligence Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .status-online {
        color: #28a745;
    }
    .status-offline {
        color: #dc3545;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def make_api_request(endpoint, method="GET", data=None):
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Please make sure the Flask server is running.")
        return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

def main():
    """Main Streamlit app"""
    
    # Header
    st.title("🤖 DevOps Intelligence Platform")
    st.markdown("---")
    
    # Check API health
    health_data = check_api_health()
    if health_data:
        st.success("✅ Connected to API")
        
        # Display agent status
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            status = "🟢 Online" if health_data['agents']['search'] else "🔴 Offline"
            st.metric("Search Agent", status)
        with col2:
            status = "🟢 Online" if health_data['agents']['resolution'] else "🔴 Offline"
            st.metric("Resolution Agent", status)
        with col3:
            status = "🟢 Online" if health_data['agents']['feedback'] else "🔴 Offline"
            st.metric("Feedback Agent", status)
        with col4:
            status = "🟢 Online" if health_data['agents']['proactive'] else "🔴 Offline"
            st.metric("Proactive Agent", status)
    else:
        st.error("❌ Cannot connect to API. Please start the Flask server first.")
        st.code("python app.py", language="bash")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Agent",
        ["Search Agent", "Resolution Agent", "Feedback Agent", "Proactive Agent", "Dashboard"]
    )
    
    if page == "Search Agent":
        search_agent_page()
    elif page == "Resolution Agent":
        resolution_agent_page()
    elif page == "Feedback Agent":
        feedback_agent_page()
    elif page == "Proactive Agent":
        proactive_agent_page()
    elif page == "Dashboard":
        dashboard_page()

def search_agent_page():
    """Search Agent interface"""
    st.header("🔍 Search Agent")
    st.markdown("Semantic search across logs, configs, and incidents")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_input("Search Query", placeholder="e.g., database timeout issues")
        k = st.slider("Number of Results", 1, 20, 5)
        
        col_search, col_load = st.columns(2)
        with col_search:
            if st.button("🔍 Search", type="primary"):
                if query:
                    with st.spinner("Searching..."):
                        data = make_api_request("search", "POST", {"query": query, "k": k})
                        if data:
                            display_search_results(data)
                else:
                    st.warning("Please enter a search query")
        
        with col_load:
            if st.button("📁 Load Incident Data"):
                with st.spinner("Loading data..."):
                    data = make_api_request("data/load", "POST")
                    if data:
                        st.success(f"✅ {data['message']}")
    
    with col2:
        st.info("💡 **Tips:**\n- Use natural language queries\n- Include specific error messages\n- Mention services or environments")

def display_search_results(data):
    """Display search results"""
    st.subheader(f"Search Results for: '{data['query']}'")
    st.write(f"Found {data['total_results']} results")
    
    for i, result in enumerate(data['results'], 1):
        with st.expander(f"{i}. {result['log_id']} (Score: {result['similarity_score']:.3f})"):
            st.write("**Content:**")
            st.write(result['content'])
            
            st.write("**Metadata:**")
            st.json(result['metadata'])
            
            if result['related_context']:
                st.write("**Related Context:**")
                st.json(result['related_context'])

def resolution_agent_page():
    """Resolution Agent interface"""
    st.header("💡 Resolution Agent")
    st.markdown("Generate actionable recommendations and action plans")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_area(
            "Issue Description", 
            placeholder="Describe the issue you need help with...",
            height=100
        )
        
        if st.button("🚀 Get Recommendations", type="primary"):
            if query:
                with st.spinner("Generating recommendations..."):
                    data = make_api_request("resolution", "POST", {"query": query})
                    if data:
                        display_resolution_results(data)
            else:
                st.warning("Please describe the issue")
    
    with col2:
        st.info("💡 **Tips:**\n- Be specific about the problem\n- Include error messages\n- Mention affected services\n- Describe symptoms")

def display_resolution_results(data):
    """Display resolution results"""
    st.subheader("🎯 Resolution Recommendations")
    
    if data['recommendations']:
        for i, rec in enumerate(data['recommendations'], 1):
            with st.expander(f"{i}. {rec['title']} (Priority: {rec['priority']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Confidence:** {rec['confidence_score']:.1%}")
                    st.write(f"**Estimated Time:** {rec['estimated_time']}")
                    st.write(f"**Description:** {rec['description']}")
                
                with col2:
                    st.write("**Steps:**")
                    for step in rec['steps']:
                        st.write(f"• {step}")
                    
                    if rec['prerequisites']:
                        st.write("**Prerequisites:**")
                        for prereq in rec['prerequisites']:
                            st.write(f"• {prereq}")
    
    if data['action_plan']:
        st.subheader("📋 Action Plan")
        plan = data['action_plan']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Title:** {plan['title']}")
            st.write(f"**Total Time:** {plan['total_estimated_time']}")
            st.write(f"**Description:** {plan['description']}")
        
        with col2:
            st.write("**Success Criteria:**")
            for criteria in plan['success_criteria']:
                st.write(f"✓ {criteria}")
        
        if plan['steps']:
            st.write("**Action Steps:**")
            for step in plan['steps']:
                st.write(f"**{step.get('order', '?')}.** {step.get('action', 'Unknown Action')}")
                st.write(f"   Time: {step.get('estimated_time', 'Unknown')}")
                st.write(f"   Risk: {step.get('risk_level', 'Unknown')}")
                st.write("---")

def feedback_agent_page():
    """Feedback Agent interface"""
    st.header("👍 Feedback Agent")
    st.markdown("Help the system learn from your interactions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_input("Query", placeholder="Enter the query you searched for")
        
        feedback_type = st.radio(
            "Was this helpful?",
            ["👍 Helpful", "👎 Not Helpful"],
            horizontal=True
        )
        
        feedback_text = st.text_area(
            "Additional Comments (Optional)",
            placeholder="Tell us more about your experience...",
            height=80
        )
        
        if st.button("📤 Submit Feedback", type="primary"):
            if query:
                with st.spinner("Submitting feedback..."):
                    data = make_api_request("feedback", "POST", {
                        "query": query,
                        "feedback_type": "thumbs_up" if feedback_type == "👍 Helpful" else "thumbs_down",
                        "feedback_value": feedback_type == "👍 Helpful",
                        "feedback_text": feedback_text
                    })
                    if data:
                        display_feedback_results(data)
            else:
                st.warning("Please enter the query you searched for")
    
    with col2:
        st.info("💡 **Your feedback helps:**\n- Improve search relevance\n- Better recommendations\n- Enhanced action plans\n- Continuous learning")

def display_feedback_results(data):
    """Display feedback results"""
    st.success("✅ Feedback submitted successfully!")
    st.write(f"**Feedback ID:** {data['feedback_id']}")
    st.write(f"**Message:** {data['message']}")
    
    if 'learning_insights' in data:
        insights = data['learning_insights']
        st.subheader("📊 Learning Insights")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Feedback", insights['total_feedback'])
        with col2:
            st.metric("Accuracy", f"{insights['accuracy']:.1%}")
        with col3:
            st.metric("Positive Feedback", insights['positive_feedback'])

def proactive_agent_page():
    """Proactive Agent interface"""
    st.header("⚠️ Proactive Agent")
    st.markdown("Monitor tickets and get early warning alerts")
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶️ Start Monitoring", type="primary"):
            with st.spinner("Starting monitoring..."):
                data = make_api_request("proactive/start", "POST")
                if data:
                    st.success(data['message'])
    
    with col2:
        if st.button("⏹️ Stop Monitoring"):
            with st.spinner("Stopping monitoring..."):
                data = make_api_request("proactive/stop", "POST")
                if data:
                    st.success(data['message'])
    
    with col3:
        if st.button("🔄 Refresh Data"):
            with st.spinner("Refreshing data..."):
                data = make_api_request("proactive/alerts")
                if data:
                    display_proactive_results(data)

def display_proactive_results(data):
    """Display proactive results"""
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Alerts", data['total_alerts'], delta=None)
    with col2:
        st.metric("Emerging Issues", data['total_issues'], delta=None)
    with col3:
        st.metric("Active Clusters", data['total_clusters'], delta=None)
    
    # Alerts
    if data['alerts']:
        st.subheader("🚨 Active Alerts")
        for alert in data['alerts']:
            with st.expander(f"{alert['title']} ({alert['severity']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Type:** {alert['alert_type']}")
                    st.write(f"**Confidence:** {alert['confidence']:.1%}")
                    st.write(f"**Created:** {alert['created_at']}")
                
                with col2:
                    st.write(f"**Message:** {alert['message']}")
                    if alert['escalation_timeframe']:
                        st.write(f"**Escalation:** {alert['escalation_timeframe']}")
                
                st.write("**Recommended Actions:**")
                for action in alert['recommended_actions']:
                    st.write(f"• {action}")
                
                if not alert['acknowledged']:
                    if st.button(f"Acknowledge {alert['alert_id']}", key=f"ack_{alert['alert_id']}"):
                        with st.spinner("Acknowledging..."):
                            ack_data = make_api_request(f"proactive/acknowledge/{alert['alert_id']}", "POST")
                            if ack_data:
                                st.success("Alert acknowledged!")
                                st.rerun()
    
    # Emerging Issues
    if data['emerging_issues']:
        st.subheader("🔍 Emerging Issues")
        for issue in data['emerging_issues']:
            with st.expander(f"{issue['title']} ({issue['severity']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Confidence:** {issue['confidence_score']:.1%}")
                    st.write(f"**Ticket Count:** {issue['ticket_count']}")
                    st.write(f"**Growth Rate:** {issue['growth_rate']:.1f} tickets/hour")
                
                with col2:
                    st.write(f"**Services:** {', '.join(issue['affected_services'])}")
                    st.write(f"**Environments:** {', '.join(issue['affected_environments'])}")
                    st.write(f"**First Detected:** {issue['first_detected']}")
                
                st.write("**Root Cause Hypotheses:**")
                for hypothesis in issue['root_cause_hypothesis']:
                    st.write(f"• {hypothesis}")
                
                st.write("**Recommended Actions:**")
                for action in issue['recommended_actions']:
                    st.write(f"• {action}")
    
    # Clusters
    if data['clusters']:
        st.subheader("📊 Active Clusters")
        for cluster in data['clusters']:
            with st.expander(f"Cluster {cluster['cluster_id']} ({cluster['cluster_size']} tickets)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Average Similarity:** {cluster['avg_similarity']:.3f}")
                    st.write(f"**Created:** {cluster['created_at']}")
                
                with col2:
                    st.write("**Common Patterns:**")
                    for pattern in cluster['common_patterns']:
                        st.write(f"• {pattern}")
                
                st.write("**Severity Distribution:**")
                st.bar_chart(cluster['severity_distribution'])
                
                st.write("**Service Distribution:**")
                st.bar_chart(cluster['service_distribution'])

def dashboard_page():
    """Dashboard page"""
    st.header("📊 Dashboard")
    st.markdown("Overview of all agents and system status")
    
    # Get data from all agents
    with st.spinner("Loading dashboard data..."):
        # Health check
        health_data = check_api_health()
        
        # Proactive data
        proactive_data = make_api_request("proactive/alerts")
        
        # Display overview
        if health_data:
            st.subheader("🔧 System Status")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                status = "🟢 Online" if health_data['agents']['search'] else "🔴 Offline"
                st.metric("Search Agent", status)
            with col2:
                status = "🟢 Online" if health_data['agents']['resolution'] else "🔴 Offline"
                st.metric("Resolution Agent", status)
            with col3:
                status = "🟢 Online" if health_data['agents']['feedback'] else "🔴 Offline"
                st.metric("Feedback Agent", status)
            with col4:
                status = "🟢 Online" if health_data['agents']['proactive'] else "🔴 Offline"
                st.metric("Proactive Agent", status)
        
        if proactive_data:
            st.subheader("📈 Proactive Intelligence")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Active Alerts", proactive_data['total_alerts'])
            with col2:
                st.metric("Emerging Issues", proactive_data['total_issues'])
            with col3:
                st.metric("Active Clusters", proactive_data['total_clusters'])
            
            # Quick actions
            st.subheader("🚀 Quick Actions")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔍 Quick Search"):
                    st.session_state.page = "Search Agent"
                    st.rerun()
            
            with col2:
                if st.button("💡 Get Resolution"):
                    st.session_state.page = "Resolution Agent"
                    st.rerun()
            
            with col3:
                if st.button("👍 Give Feedback"):
                    st.session_state.page = "Feedback Agent"
                    st.rerun()
            
            with col4:
                if st.button("⚠️ View Alerts"):
                    st.session_state.page = "Proactive Agent"
                    st.rerun()

if __name__ == "__main__":
    main()
