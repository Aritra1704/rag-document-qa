# 🤖 Social Media AI Agent

An AI-powered social media post generator with **human-in-the-loop** approval workflow. Generate engaging posts, review them, and publish automatically to Twitter/X.

## ✨ Features

- **AI Post Generation**: Uses GPT-4 or Claude to create platform-optimized content
- **Human-in-the-Loop**: Review, edit, or reject posts before publishing
- **Platform**: Support for Twitter/X
- **MCP Integration**: Use as an MCP server with AI assistants
- **Web Interface**: Streamlit UI for easy approval workflow
- **Railway Ready**: Deploy to Railway with one click

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  User Request   │───▶│  LangGraph Agent │───▶│ Generate Post   │
└─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                        │
                       ┌──────────────────┐    ┌────────▼────────┐
                       │  Publish Post    │◀───│ Human Approval  │
                       └────────┬─────────┘    │   (interrupt)   │
                                │              └─────────────────┘
                       ┌────────▼─────────┐
                       │     Twitter/X    │
                       └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key or Anthropic API key
- Twitter/X Developer credentials (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/social-media-ai-agent.git
cd social-media-ai-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Running Locally

```bash
# Run both API and UI
python -m src.main all

# Or run separately:
python -m src.main api  # FastAPI on http://localhost:8000
python -m src.main ui   # Streamlit on http://localhost:8501
python -m src.main mcp  # MCP server for AI assistants
```

### Using the Web UI

1. Open http://localhost:8501 in your browser
2. Enter a topic in the sidebar
3. Select tone
4. Click "Generate Post"
5. Review the generated post
6. Click Approve, Reject, or Edit

## 🔧 Configuration

Set these environment variables in your `.env` file:

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes* |
| `ANTHROPIC_API_KEY` | Anthropic API key | Yes* |
| `LLM_PROVIDER` | `openai` or `anthropic` | No (default: openai) |
| `LLM_MODEL` | Model name (e.g., `gpt-4o`) | No |
| `TWITTER_API_KEY` | Twitter API key | For Twitter |
| `TWITTER_API_SECRET` | Twitter API secret | For Twitter |
| `TWITTER_ACCESS_TOKEN` | Twitter access token | For Twitter |
| `TWITTER_ACCESS_SECRET` | Twitter access secret | For Twitter |


*At least one LLM key required

## 🚂 Deploy to Railway

1. Push this repo to GitHub
2. Go to [Railway](https://railway.app)
3. Create new project → Deploy from GitHub repo
4. Add environment variables in Railway dashboard
5. Deploy!

Your API will be available at `https://your-app.railway.app`

### Environment Variables in Railway

Add all variables from `.env.example` in the Railway dashboard under:
**Variables** → **Add Variable**

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/posts/generate` | Generate a new post |
| `GET` | `/posts/pending` | List pending approvals |
| `POST` | `/posts/{id}/approve` | Approve/reject/edit post |
| `GET` | `/posts/{id}` | Get post status |

### Example: Generate a Post

```bash
curl -X POST http://localhost:8000/posts/generate \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "AI in healthcare",
    "platform": "twitter",
    "tone": "professional"
  }'
```

## 🔌 MCP Integration

Use this agent with Claude Desktop or other MCP clients:

```json
{
  "mcpServers": {
    "social-media-agent": {
      "command": "python",
      "args": ["-m", "src.main", "mcp"],
      "cwd": "/path/to/social-media-ai-agent"
    }
  }
}
```

### Available MCP Tools

- `generate_social_post` - Generate a new post
- `list_pending_posts` - List posts awaiting approval
- `approve_post` - Approve a pending post
- `reject_post` - Reject and regenerate
- `edit_and_approve_post` - Edit and publish

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines first.

---

Built with ❤️ using LangGraph, FastAPI, and Streamlit
