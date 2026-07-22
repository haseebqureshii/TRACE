# Drift-Free Customer Service Agent Platform

## Overview

The Drift-Free Customer Service Agent Platform is an AI-powered customer support system designed to provide accurate, context-aware responses while maintaining strict boundaries around supported topics. The platform uses a knowledge base-driven approach to ensure all responses are grounded in verified business information, preventing hallucinations or ungrounded policy statements.

## Key Features

- **Context-Aware Conversations**: The platform maintains conversation history and resolves pronouns across multiple turns to provide coherent, contextually relevant responses.
- **Knowledge Base Retrieval**: Uses vector embeddings to retrieve relevant business documentation (return policies, shipping times, store hours, payment methods, etc.) for accurate response generation.
- **Drift Detection & Escalation**: Monitors user queries for off-topic or unsupported requests. After a configurable number of off-topic queries, the system automatically escalates to a human support specialist.
- **Output Grounding Validation**: Ensures AI-generated responses strictly adhere to the provided knowledge base context without inventing ungrounded policies.

## How to Use

### Getting Started

1. **Upload a Knowledge Base File**: In the sidebar, upload a valid `.json` Knowledge Base file containing your business information, policies, and frequently asked questions.

2. **Automatic Session Initialization**: Once a valid Knowledge Base file is uploaded, the system automatically initializes a support session tailored to your business.

3. **Start Chatting**: Type your customer support inquiries in the chat input area. The AI assistant will provide responses based on your knowledge base.

4. **Multi-Turn Conversations**: The system maintains conversation context, allowing you to ask follow-up questions with pronouns or references to previous messages.

### Knowledge Base JSON File Structure

The Knowledge Base JSON file is the foundation of your customer support agent. It must follow this rigid structure:

```json
{
  "business_name": "Your Business Name",
  "domain_description": "Brief description of your business domain and what support topics are covered",
  "documents": [
    {
      "id": "doc_001",
      "category": "Returns & Refunds",
      "title": "Return Policy for Opened Items",
      "content": "Items that have been opened may be returned within 30 days of purchase... [full policy text]"
    },
    {
      "id": "doc_002",
      "category": "Shipping",
      "title": "Shipping Times and Methods",
      "content": "Standard shipping takes 3-5 business days... [full shipping information]"
    },
    {
      "id": "doc_003",
      "category": "Store Hours",
      "title": "Customer Service Hours",
      "content": "Our customer service team is available Monday-Friday, 9 AM to 6 PM EST... [full hours information]"
    }
  ]
}
```

### Required Fields

- **`business_name`** (string): The name of your business or organization.
- **`domain_description`** (string): A brief description of what topics and support areas the AI agent is authorized to handle.
- **`documents`** (array of objects): A list of knowledge base documents. Each document must include:
  - **`id`** (string): A unique identifier for the document.
  - **`category`** (string): The topic category (e.g., "Returns & Refunds", "Shipping", "Store Hours").
  - **`title`** (string): A descriptive title for the document.
  - **`content`** (string): The full text content of the policy or information.

### File Requirements

- **Format**: Must be a valid `.json` file.
- **Size**: Maximum file size is 10 MB.
- **Structure**: Must include the top-level keys `business_name`, `domain_description`, and `documents`. Each document in the `documents` array must include `id`, `category`, `title`, and `content` fields.

## System Behavior

### In-Domain Queries
When a user asks a question that matches the knowledge base content, the system retrieves relevant documents and generates an accurate, grounded response.

### Off-Topic Queries
If a user asks a question outside the scope of the knowledge base, the system will:
1. Respond with: "I am only able to assist with customer service and support inquiries. Could you please clarify your request?"
2. Increment a hidden "drift strike" counter.

### Human Escalation
After reaching the maximum number of drift strikes (configurable, default is 3), the system will:
1. Display: "Chat diverted/escalated to a human."
2. Disable further chat input, indicating the AI session has ended and a human support specialist should take over.

## Support

For technical support or questions about the platform, please contact your system administrator or refer to the internal documentation.