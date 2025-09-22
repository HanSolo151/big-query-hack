"""
Streamlit UI for DevOps Intelligence Platform
Extended with Explainability + Multimodal results
"""

import streamlit as st
import requests
import json
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

# ---- Utility functions ----
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def make_api_request(endpoint, method="GET", data=None):
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            return None

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Please make sure the Flask server is running.")
        return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

# ---- Pages ----
def main():
    st.title("🤖 DevOps Intelligence Platform")
    st.markdown("---")

    health_data = check_api_health()
    if health_data:
        st.success("✅ Connected to API")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Search Agent", "🟢" if health_data['agents']['search'] else "🔴")
        with col2:
            st.metric("Resolution Agent", "🟢" if health_data['agents']['resolution'] else "🔴")
        with col3:
            st.metric("Feedback Agent", "🟢" if health_data['agents']['feedback'] else "🔴")
        with col4:
            st.metric("Proactive Agent", "🟢" if health_data['agents']['proactive'] else "🔴")
    else:
        st.error("❌ API not reachable. Start Flask with `python run_local.py`.")
        return

    # Sidebar nav
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose Page", 
        ["Search Agent", "Resolution Agent", "Feedback Agent", "Proactive Agent", "Dashboard"])

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
    st.header("🔍 Search Agent")
    query = st.text_input("Search Query", placeholder="e.g., database timeout issues")
    k = st.slider("Number of Results", 1, 20, 5)

    if st.button("Run Search"):
        if query:
            with st.spinner("Searching..."):
                data = make_api_request("search", "POST", {"query": query, "k": k})
                if data:
                    display_search_results(data)
        else:
            st.warning("Please enter a query.")


def display_search_results(data):
    st.subheader(f"Search Results for: '{data['query']}'")
    st.caption(f"Found {data['total_results']} results")
    for i, r in enumerate(data['results'], 1):
        with st.expander(f"{i}. {r['log_id']} (Score: {r['similarity_score']:.3f})"):
            st.write("**Content:**")
            st.write(r['content'])
            st.write("**Metadata:**")
            st.json(r['metadata'])
            if r['related_context']:
                st.write("**Related Context:**")
                st.json(r['related_context'])


def resolution_agent_page():
    st.header("🛠 Resolution Agent")
    query = st.text_area("Describe the issue", placeholder="e.g., pod crashlooping in Kubernetes")

    if st.button("Get Recommendations"):
        if query:
            with st.spinner("Running full pipeline..."):
                data = make_api_request("resolution", "POST", {"query": query})
                if data:
                    display_resolution_results(data)
        else:
            st.warning("Please describe your issue.")


def display_resolution_results(data):
    st.subheader("Recommendations")
    if data['recommendations']:
        for i, rec in enumerate(data['recommendations'], 1):
            with st.expander(f"{i}. {rec['title']} (Priority: {rec['priority']}, Confidence: {rec['confidence_score']:.2f})"):
                st.write(rec['description'])
                st.markdown("**Steps:**")
                for s in rec['steps']:
                    st.markdown(f"- {s}")
                if rec['prerequisites']:
                    st.markdown("**Prerequisites:** " + ", ".join(rec['prerequisites']))
    else:
        st.info("No recommendations found.")

    if data['action_plan']:
        st.subheader("📋 Action Plan")
        plan = data['action_plan']
        st.markdown(f"**{plan['title']}** — {plan['description']}")
        st.markdown(f"Estimated Time: {plan['total_estimated_time']}")
        for s in plan['steps']:
            st.markdown(f"- ({s['order']}) {s['action']}: {s['description']} (⏱ {s['estimated_time']})")

    if data['explainability']:
        st.subheader("🔎 Explainability")
        exp = data['explainability']
        st.write(f"**Overall Confidence:** {exp['overall_confidence']}%")
        st.write(f"**Reasoning:** {exp['reasoning']}")
        st.write(f"**Transparency Score:** {exp['transparency_score']}")
        with st.expander("Evidence Tickets"):
            for t in exp['evidence_tickets']:
                st.markdown(f"- {t['ticket_id']} | {t['title']} (Confidence {t['confidence_percentage']}%)")

    if data['multimodal_results']:
        st.subheader("🖼 Multimodal Analysis")
        for mm in data['multimodal_results']:
            with st.expander(f"{mm['title']} ({mm['content_type']}, Score: {mm['similarity_score']:.3f})"):
                if mm.get("visual_description"):
                    st.write("**Visual Description:**")
                    st.write(mm['visual_description'])
                if mm.get("matched_content"):
                    st.write("**Matched Content:**")
                    st.write(mm['matched_content'])
                st.json(mm['metadata'])


def feedback_agent_page():
    st.header("💬 Feedback Agent")
    query = st.text_input("Feedback Query")
    fb_type = st.selectbox("Feedback Type", ["thumbs_up", "thumbs_down", "rating", "natural_language"])
    fb_value = st.text_input("Feedback Value", "👍")
    fb_text = st.text_area("Additional Notes")

    if st.button("Submit Feedback"):
        with st.spinner("Submitting feedback..."):
            data = make_api_request("feedback", "POST", {
                "query": query,
                "feedback_type": fb_type,
                "feedback_value": fb_value,
                "feedback_text": fb_text
            })
            if data:
                st.success(f"Feedback ID: {data['feedback_id']}")
                st.json(data['learning_insights'])


def proactive_agent_page():
    st.header("🚨 Proactive Monitoring")
    if st.button("Refresh Alerts"):
        with st.spinner("Fetching proactive alerts..."):
            data = make_api_request("proactive/alerts")
            if data:
                st.subheader("Active Alerts")
                for a in data['alerts']:
                    st.markdown(f"- [{a['alert_type']}] {a['title']} ({a['severity']}, conf {a['confidence']})")
                st.subheader("Emerging Issues")
                st.json(data['issues'])


def dashboard_page():
    st.header("📊 System Dashboard")
    health_data = check_api_health()
    if health_data:
        st.json(health_data)
    else:
        st.error("Health data unavailable.")


if __name__ == "__main__":
    main()
