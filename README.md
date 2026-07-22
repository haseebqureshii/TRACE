# Drift-Free Customer Service Agent Platform

This project is a Drift-Free Customer Service Agent platform that provides guardrails for AI customer support agents, including query contextualization, knowledge base retrieval, output grounding validation, and drift-strike escalation.

## Setup Instructions

### 1. Virtual Environment Setup

A Python virtual environment is required for this project. It has been created in the `.venv` directory.

**To activate the virtual environment:**

- **Windows (PowerShell or Command Prompt):**
  ```powershell
  .venv\Scripts\activate
  ```

- **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Dependencies

With the virtual environment activated, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the `.env` file and update it with your actual API key and LLM endpoint:

```env
TRACE_LLM_API_KEY_TEST=your_key_here
TRACE_LLM_BASE_URL=http://your-university-llm-endpoint/v1
TEST_MODEL_ID=qwen3-235b-a22b-instruct-2507
```

### 4. Verify Connection

To verify that the LLM connection is working correctly, run the connection test:

```bash
python -m tests.test_connection
```

## Running the Application

### a. Running Locally

**Start the Backend (FastAPI):**

```bash
uvicorn src.api:app --reload
```

The backend will be available at `http://127.0.0.1:8000`. You can access the API documentation at `http://127.0.0.1:8000/docs`.

**Start the Frontend (Streamlit):**

In a new terminal (with the virtual environment activated), run:

```bash
streamlit run streamlit_app.py
```

The Streamlit app will be available at `http://localhost:8501`.

### b. Hybrid Public Deployment (Streamlit Cloud + Local Ngrok Tunnel)

For public access while keeping the backend on a local or university network:

1. **Start the local FastAPI server** while connected to the university VPN:
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```

2. **Expose port 8000 via ngrok:**
   ```bash
   ngrok http 8000
   ```
   This will generate a public URL like `https://xxxxx.ngrok-free.app`.

3. **Deploy `streamlit_app.py` to Streamlit Community Cloud:**
   - Push your code to a GitHub repository.
   - Go to [Streamlit Community Cloud](https://share.streamlit.io/) and connect your repository.
   - Set the main file to `streamlit_app.py`.

4. **Configure the ngrok URL in the Streamlit app:**
   - Open the deployed Streamlit app.
   - In the sidebar, expand "⚙️ Advanced Settings".
   - Enter the ngrok public URL (e.g., `https://xxxxx.ngrok-free.app`) in the Backend URL field.
   - The app will automatically verify the connection and display the system status badge.

## Project Structure

```
├── config/
│   └── rails/           # For NeMo / Colang rules (if applicable)
├── src/
│   ├── __init__.py
│   ├── api.py           # FastAPI backend with chat and session endpoints
│   ├── llm_client.py    # OpenAI-compatible client setup
│   ├── pipeline.py      # Chat turn processing pipeline
│   ├── state.py         # SessionState and SessionManager
│   └── guardrails/      # Custom guardrail modules (contextualizer, input_rail, evaluator)
├── tests/
│   ├── __init__.py
│   ├── test_api.py      # FastAPI endpoint tests
│   ├── test_pipeline.py # Pipeline logic tests
│   └── fixtures/        # Test fixtures (kb_documents.json, test_conversations.json, evaluator_config.json)
├── .env
├── .gitignore
├── requirements.txt
├── streamlit_app.py     # Streamlit frontend application
└── README.md
```

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /api/v1/session/init` - Initialize a new session with KB data
- `POST /api/v1/chat` - Process a chat turn
- `GET /api/v1/session/{session_id}` - Get session state
- `POST /api/v1/session/reset` - Reset a session