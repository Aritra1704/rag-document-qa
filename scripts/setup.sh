#!/bin/bash
# Setup script for RAG Document Q&A System

set -e

echo "🚀 Setting up RAG Document Q&A System..."

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Python $required_version or higher is required. You have $python_version"
    exit 1
fi
echo "✅ Python version: $python_version"

# Create virtual environment
echo "🔨 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt --quiet
echo "✅ Dependencies installed"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs chroma_db
echo "✅ Directories created"

# Copy .env.example to .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created - Please add your ANTHROPIC_API_KEY"
else
    echo "✅ .env file already exists"
fi

# Run tests
echo "🧪 Running tests..."
pip install -r requirements-dev.txt --quiet
pytest tests/ -v

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Edit .env file and add your ANTHROPIC_API_KEY"
echo "2. Run: source venv/bin/activate"
echo "3. Run: streamlit run src/app.py"
echo ""
echo "🎉 Happy coding!"
