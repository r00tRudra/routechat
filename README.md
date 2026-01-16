# Nova Chat - Multi-LLM Router Chatbot

A beautiful, minimalistic web-based chatbot that intelligently routes user prompts to different Groq models based on the type of question.

## Features

- Smart Model Routing — automatically chooses the best model based on prompt content  
- Supports 4 different Grok models with specialized roles
- Modern dark-themed chat UI with auto-growing input
- Real-time responses via WebSocket (Socket.IO)
- Hover toolbars, copy/edit/regenerate/love/dislike buttons (frontend ready)
- Generation time display
- Scroll-to-bottom button

## Currently Supported Models & Routing Logic

| Key     | Model ID                                      | Purpose / Specialty                              |
|---------|-----------------------------------------------|--------------------------------------------------|
| model_xl| openai/gpt-oss-safeguard-20b                  | Safety analysis, policy-sensitive content        |
| model_l | meta-llama/llama-4-maverick-17b-128e-instruct | Complex reasoning, system design, deep answers   |
| model_m | meta-llama/llama-guard-4-12b                  | Cybersecurity, exploits, moderation              |
| model_s | groq/compound-mini                            | Casual chat, greetings, simple & fast questions  |

Routing uses sentence-transformers + cosine similarity (very lightweight)

## Tech Stack

- Backend: Flask + Flask-SocketIO
- LLM: Groq API
- Embedding/Routing: sentence-transformers + scikit-learn
- Frontend: Pure HTML/CSS + JavaScript + Socket.IO
- Environment: python-dotenv

## Quick Start

1. Clone the repository
```bash
git clone https://github.com/yourusername/nova-chat.git
cd nova-chat