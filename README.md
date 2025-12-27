# AI-Chatbot-Mentor

### 🤖 AI Chatbot Mentor — Domain-Specific Learning Assistant (LangChain + Streamlit)

This project implements an interactive AI-powered learning mentor that allows users to select a specific learning domain (such as Python, SQL, Power BI, Machine Learning, Generative AI, etc.) and ask topic-focused questions within that module.

The goal of this project was not just to build a chatbot — but to understand how real-world LLM-driven learning systems are designed, including:

- context-aware conversation memory

- prompt-based domain control

- module-wise workflows

- multi-model integration using LangChain

This project helped me deeply understand the complete workflow of building an AI mentor system — from UI interactions and session state handling to LLM chaining, prompt engineering, and response filtering.

The intention of the project was learning-oriented design & implementation rather than building a fully autonomous assistant, and it successfully served that purpose.

### Project Overview

The idea is simple:

Select a learning module → Ask questions → Receive responses only within that domain.

Instead of responding like a generic chatbot, the system:

✔ restricts responses to the selected subject
✔ uses structured prompts to maintain topic boundaries
✔ leverages conversation memory for continuity
✔ supports multiple LLM models based on module type

To achieve this, I implemented the workflow manually using LangChain, rather than relying on pre-built chatbot wrappers — which helped me understand the underlying mechanics more clearly.

### 🔄 End-to-End Workflow
**Module Selection**

User chooses a learning domain such as:

- Python
- SQL
- Power BI
- EDA
- Machine Learning
- Deep Learning
- Generative AI
- Agentic AI

The UI switches into a dedicated module chat interface.

### 🧵 Conversation Memory

A custom memory class stores:

user queries

AI responses

session-wise history

The memory is:

persisted inside Streamlit session state

passed to the LLM as conversational context

reset when switching modules

This helped me understand how memory systems work in LLM apps.

### LLM Integration

Multiple models are used depending on the module:

Gemini 2.5 Flash — Python / ML / DL topics

Hugging Face Endpoints — SQL / GenAI / EDA / Power BI / Agentic AI

Models are wrapped using:

HuggingFaceEndpoint

ChatHuggingFace

This allowed me to explore:

✔ model routing
✔ endpoint execution
✔ performance differences
✔ domain-specific response behavior

### 🧾 Prompt Engineering

Instead of keyword filtering, the system uses instruction-based soft constraints, meaning:

relevant cross-domain topics are allowed

unrelated questions are politely declined

responses remain educational and concise

This helped me understand how prompt discipline improves reliability compared to hard-rule text filters.

### 💬 Chat Interface

The UI includes:

chat-style message bubbles with user & assistant roles

persistent chat history display

downloadable conversation logs

session end & module reset options

The goal was to create a mentor-like interaction experience rather than a plain text chatbot.

### 🧠 Model Logic & Architecture

The application follows:

Module Selection
→ Context-aware prompt
→ LLM response
→ Memory update
→ UI chat display

Key components include:

CustomConversationMemory

LangChain chaining pipeline

Structured prompt templates

Session-based UI state management

This project strengthened my understanding of:

✔ LangChain Runnable pipelines
✔ Memory-driven conversation flows
✔ Multi-model orchestration
✔ UX design for AI systems

### 🧰 Tech Stack
- Python
- Streamlit
- LangChain
- Gemini 2.5 Flash
- HuggingFace Models
- dotenv for API key management

### ✅ Outcomes & Learnings

Although the project is still evolving, it helped me:

✔ Build a structured LLM application end-to-end
✔ Understand conversation memory design
✔ Implement multi-module learning workflows
✔ Improve prompt alignment & response control
✔ Design a clean, learner-focused chatbot interface

This serves as a strong foundation for future AI mentor & tutoring systems.

### 🚀 Future Enhancements

- Some upcoming improvements I plan to explore:
- Knowledge-grounded answers using vector DBs
- User learning progress tracking
- Topic-wise quizzes & explanations
- Adaptive hints and follow-up guidance
- Fine-tuned domain mentor models

### 📬 Contact

If you’d like to review the project, suggest improvements, or collaborate — I’d be happy to connect!

📧 Email — rohithmasineni223@email.com

🔗 LinkedIn — Rohith Kumar Masineni

⭐ If you find this project useful, feel free to star the repository!
