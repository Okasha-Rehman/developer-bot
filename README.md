# 🤖 AI Code Assistant with RAG + Context7

> A smart developer chatbot that understands YOUR codebase and official documentation

Ask questions about your code and get answers grounded in both your local files (via RAG) and official documentation (via Context7 API). No more context-switching between your IDE, docs, and ChatGPT

## ✨ Features

- 🔍 **RAG (Retrieval-Augmented Generation)**: Indexes your local codebase using FAISS vector database
- 📚 **Context7 Integration**: Fetches official documentation from Context7 API
- 🧠 **Dual Context**: Combines local code knowledge with official docs for comprehensive answers
- 💬 **Interactive Chat**: Beautiful Streamlit interface with session management
- ⚡ **Fast Search**: FAISS provides millisecond similarity search
- 📊 **Source Tracking**: See which sources (RAG/Context7) were used for each answer

## 🎯 Use Cases

- **Understand Legacy Code**: "What does the authentication flow do?"
- **Learn Best Practices**: "How should I structure my FastAPI routes?"
- **Debug Issues**: "Where is the database connection initialized?"
- **Compare Implementations**: "Is my error handling following FastAPI standards?"
- **Quick Documentation**: "What files handle user sessions?"

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Groq API Key](https://console.groq.com/keys) (free)
- [Context7 API Key](https://context7.com) (free)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Okasha-Rehman/developer-bot
cd ai-code-assistant
```

2. **Create virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

3. **Install dependencies**
```bash
uv sync
```

4. **Configure environment variables**
```bash
# Create .env file
cp .env.example .env

# Edit .env and add your API keys
GROQ_API_KEY=your_groq_api_key_here
CONTEXT7_API_KEY=your_context7_key_here
```

### Running the Application

**Terminal 1 - Backend:**
```bash
uvicorn backend:app --reload
```

**Terminal 2 - Frontend:**
```bash
streamlit run frontend.py
```

## 📖 How to Use

### Index Your Code

1. Click **"Index Code"** in the sidebar
2. Enter the path to your codebase (or use `.` for current directory)
3. Wait for indexing to complete (1-2 minutes for medium projects)
4. See confirmation: "Indexed X files, Y chunks"

### Check Sources

Each response shows which sources were used:
- `RAG: ✅` = Used your local code
- `Context7: ✅` = Used official documentation
- Both ✅ = Combined both sources

⭐ Star this repo if you found it helpful!