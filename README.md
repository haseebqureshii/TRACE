# TRACE: AI Language Tutor Guardrails Research Project

This project is a research initiative focused on developing and evaluating guardrails for AI Language Tutors.

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

## Project Structure

```
├── config/
│   └── rails/           # For NeMo / Colang rules
├── src/
│   ├── __init__.py
│   ├── llm_client.py    # OpenAI-compatible client setup using TRACE_LLM_API_KEY_TEST
│   └── guardrails/      # Custom anti-drift modules
├── tests/
│   ├── __init__.py
│   └── test_connection.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md