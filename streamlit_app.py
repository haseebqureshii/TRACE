import os
import json
import requests
import streamlit as st

# Configuration
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
MAX_FILE_SIZE_MB = 5
HEALTH_CHECK_TIMEOUT = 3
MAX_MESSAGE_LENGTH = 1000


def get_backend_url():
    """Get backend URL from Streamlit secrets or environment variable."""
    # Try to get from Streamlit secrets first
    backend_url = st.secrets.get("BACKEND_URL") if hasattr(st, "secrets") else None
    
    if not backend_url:
        backend_url = os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL)
    
    return backend_url.rstrip("/")


def check_backend_health(backend_url: str) -> bool:
    """Check if backend is healthy by querying the /health endpoint."""
    try:
        response = requests.get(f"{backend_url}/health", timeout=HEALTH_CHECK_TIMEOUT)
        return response.status_code == 200 and response.json().get("status") == "ok"
    except Exception:
        return False


def validate_kb_file(uploaded_file) -> tuple[bool, str, dict | None]:
    """Validate KB JSON file: extension, size, syntax, and required keys."""
    # Check file extension
    if not uploaded_file.name.endswith(".json"):
        return False, "File must be a .json file.", None
    
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"File size must be under {MAX_FILE_SIZE_MB}MB.", None
    
    # Read and parse JSON
    try:
        content = uploaded_file.read().decode("utf-8")
        data = json.loads(content)
    except json.JSONDecodeError:
        return False, "Invalid JSON syntax. Please check the file format.", None
    except Exception as e:
        return False, f"Error reading file: {str(e)}", None
    
    # Check required top-level keys
    required_keys = ["business_name", "domain_description", "documents"]
    for key in required_keys:
        if key not in data:
            return False, f"Missing required key: '{key}'.", None
    
    # Validate documents structure
    if not isinstance(data.get("documents"), list):
        return False, "'documents' must be a list.", None
    
    for i, doc in enumerate(data["documents"]):
        if not isinstance(doc, dict):
            return False, f"Document at index {i} must be an object.", None
        required_doc_keys = ["id", "category", "title", "content"]
        for doc_key in required_doc_keys:
            if doc_key not in doc:
                return False, f"Document at index {i} missing required key: '{doc_key}'.", None
    
    return True, "File validated successfully.", data


def initialize_session(backend_url: str, kb_data: dict) -> tuple[bool, str, dict | None]:
    """Initialize session by sending KB data to the backend."""
    try:
        response = requests.post(f"{backend_url}/api/v1/session/init", json={"kb_data": kb_data}, timeout=10)
        if response.status_code == 200:
            result = response.json()
            return True, "Session initialized successfully.", result
        else:
            return False, "Failed to initialize session.", None
    except Exception:
        return False, "Network error during session initialization.", None


def send_chat_message(backend_url: str, session_id: str, message: str) -> tuple[bool, dict | None]:
    """Send a chat message to the backend."""
    try:
        response = requests.post(f"{backend_url}/api/v1/chat", json={"session_id": session_id, "message": message}, timeout=15)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"response": f"Error: {response.status_code} - {response.text}", "is_escalated": False, "strikes": 0, "rationale": "Backend error"}
    except Exception as e:
        return False, {"response": f"Network error: {str(e)}", "is_escalated": False, "strikes": 0, "rationale": "Network failure"}


def reset_session(backend_url: str, session_id: str):
    """Reset the session on the backend."""
    try:
        requests.post(f"{backend_url}/api/v1/session/reset", json={"session_id": session_id}, timeout=5)
    except Exception:
        pass


# Initialize session state
if "backend_url" not in st.session_state:
    st.session_state.backend_url = get_backend_url()

if "backend_online" not in st.session_state:
    st.session_state.backend_online = check_backend_health(st.session_state.backend_url)

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "business_name" not in st.session_state:
    st.session_state.business_name = None

if "session_status" not in st.session_state:
    st.session_state.session_status = "Not Initialized"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "escalated" not in st.session_state:
    st.session_state.escalated = False

if "kb_file_uploaded" not in st.session_state:
    st.session_state.kb_file_uploaded = False

if "kb_file_validated" not in st.session_state:
    st.session_state.kb_file_validated = False

if "kb_data" not in st.session_state:
    st.session_state.kb_data = None

if "init_error" not in st.session_state:
    st.session_state.init_error = None


# Sidebar
st.sidebar.title("Customer Support Agent")

# System Status Badge
if st.session_state.backend_online:
    st.sidebar.success("🟢 **System Online**\n\nBackend reachable and operational.")
else:
    st.sidebar.error("🔴 **System Offline**\n\nBackend unreachable.")

# KB File Uploader
st.sidebar.subheader("Knowledge Base")
uploaded_kb_file = st.sidebar.file_uploader("Upload KB JSON file", type=["json"], max_upload_size=MAX_FILE_SIZE_MB)

# Handle file upload and validation
if uploaded_kb_file:
    is_valid, validation_msg, kb_data = validate_kb_file(uploaded_kb_file)
    if is_valid:
        st.session_state.kb_file_uploaded = True
        st.session_state.kb_file_validated = True
        st.session_state.kb_data = kb_data
        st.sidebar.success(f"File validated: {validation_msg}")
        
        # Automatically initialize session if backend is online and session not yet initialized
        if st.session_state.backend_online and not st.session_state.session_id:
            with st.spinner("Initializing session..."):
                try:
                    success, msg, result = initialize_session(st.session_state.backend_url, kb_data)
                    if success:
                        st.session_state.session_id = result["session_id"]
                        st.session_state.business_name = result["business_name"]
                        st.session_state.session_status = "Active"
                        st.session_state.chat_history = []
                        st.session_state.escalated = False
                        st.session_state.init_error = None
                        st.sidebar.success("Session initialized successfully!")
                    else:
                        st.session_state.init_error = "Failed to initialize session. Please try again."
                        st.sidebar.error("Failed to initialize session. Please try again.")
                        st.session_state.kb_file_validated = False
                        st.session_state.kb_file_uploaded = False
                        st.session_state.kb_data = None
                except Exception:
                    st.session_state.init_error = "Failed to initialize session. Please try again."
                    st.sidebar.error("Failed to initialize session. Please try again.")
                    st.session_state.kb_file_validated = False
                    st.session_state.kb_file_uploaded = False
                    st.session_state.kb_data = None
        elif not st.session_state.backend_online:
            st.sidebar.info("Backend is offline. Cannot initialize session.")
    else:
        st.sidebar.error(f"Validation error: {validation_msg}")
        st.session_state.kb_file_uploaded = False
        st.session_state.kb_file_validated = False
        st.session_state.kb_data = None
else:
    st.sidebar.info("Upload a valid .json Knowledge Base file to automatically start a session.")
    st.session_state.kb_file_uploaded = False
    st.session_state.kb_file_validated = False
    st.session_state.kb_data = None

# Display initialization error if any
if st.session_state.init_error:
    st.error(st.session_state.init_error)

# Active Session Info
st.sidebar.subheader("Active Session")
if st.session_state.session_id:
    st.sidebar.info(f"**Business:** {st.session_state.business_name or 'N/A'}\n\n**Status:** {st.session_state.session_status}\n\n**Session ID:** `{st.session_state.session_id[:8]}...`")
    if st.sidebar.button("Reset Session"):
        reset_session(st.session_state.backend_url, st.session_state.session_id)
        st.session_state.session_id = None
        st.session_state.business_name = None
        st.session_state.session_status = "Not Initialized"
        st.session_state.chat_history = []
        st.session_state.escalated = False
        st.session_state.kb_file_uploaded = False
        st.session_state.kb_file_validated = False
        st.session_state.kb_data = None
        st.session_state.init_error = None
        st.rerun()
else:
    st.sidebar.info("No active session.")

# Main Chat View
st.title("AI Customer Support Agent")

# Offline warning banner
if not st.session_state.backend_online:
    st.warning("⚠️ **The AI Customer Support system is currently offline for maintenance. Please check back later.**")

# Render chat history
for message in st.session_state.chat_history:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.write(content)

# Chat input and submission
if st.session_state.backend_online and not st.session_state.escalated and st.session_state.session_id:
    user_input = st.chat_input("Type your message... (max 1000 characters)")
    
    if user_input:
        # Enforce character limit
        if len(user_input) > MAX_MESSAGE_LENGTH:
            st.error(f"Message must be {MAX_MESSAGE_LENGTH} characters or less. Please shorten your message.")
            st.rerun()
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Render user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Send to backend
        with st.spinner("Assistant is typing..."):
            success, result = send_chat_message(st.session_state.backend_url, st.session_state.session_id, user_input)
        
        # Process response
        response_text = result.get("response", "No response received.")
        is_escalated = result.get("is_escalated", False)
        
        # Render assistant response
        with st.chat_message("assistant"):
            st.write(response_text)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        
        # Check for escalation
        if is_escalated or "Chat diverted/escalated to a human." in response_text:
            st.session_state.escalated = True
            st.session_state.session_status = "Escalated to Human"
            st.info("ℹ️ This AI session has ended. Your request has been escalated to a human support specialist.")
else:
    if not st.session_state.backend_online:
        st.chat_input("System is offline", disabled=True)
    elif st.session_state.escalated:
        st.chat_input("Session escalated to human support", disabled=True)
    elif not st.session_state.session_id:
        st.chat_input("Upload a valid Knowledge Base .json file to start...", disabled=True)