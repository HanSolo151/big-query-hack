"""
Local Development Runner for DevOps Intelligence Platform
Starts both Flask API and Streamlit UI
"""

import subprocess
import sys
import time
import threading
import webbrowser
from pathlib import Path

def run_flask():
    """Run Flask API server"""
    print(" Starting Flask API server...")
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print(" Flask server stopped")
    except Exception as e:
        print(f" Flask server error: {e}")

def run_streamlit():
    """Run Streamlit UI"""
    print(" Starting Streamlit UI...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8502",
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print(" Streamlit server stopped")
    except Exception as e:
        print(f" Streamlit server error: {e}")

def main():
    """Main runner function"""
    print(" DevOps Intelligence Platform - Local Development")
    print("=" * 60)
    
    # Check if required files exist
    required_files = ["app.py", "streamlit_app.py", "SEARCH_AGENT_1.py", "RESOLUTION_AGENT_.py", "FEEDBACK_INTEGRATION_AGENT.py", "PROACTIVE_AGENT.py"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f" Missing required files: {missing_files}")
        return
    
    print(" All required files found")
    print("\n Starting services:")
    print("   • Flask API: http://localhost:5000")
    print("   • Streamlit UI: http://localhost:8502")
    print("   • Bootstrap UI: http://localhost:5000 (Flask template)")
    print("\n Tips:")
    print("   • Make sure your API keys and credentials are configured")
    print("   • Press Ctrl+C to stop all services")
    print("   • Check the terminal for any initialization errors")
    print("\n" + "=" * 60)
    
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Wait a bit for Flask to start
    time.sleep(3)
    
    # Open browser to Streamlit
    try:
        webbrowser.open("http://localhost:8502")
    except:
        pass
    
    # Start Streamlit (this will block)
    try:
        run_streamlit()
    except KeyboardInterrupt:
        print("\n Shutting down all services...")
        print(" All services stopped")

if __name__ == "__main__":
    main()
