"""Main entry point for the Social Media AI Agent."""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path


def run_api():
    """Run the FastAPI server."""
    import uvicorn

    from src.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "src.web.api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


def run_ui():
    """Run the Streamlit UI."""
    ui_path = Path(__file__).parent / "web" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)])


def run_mcp():
    """Run the MCP server."""
    from src.mcp.server import main

    asyncio.run(main())


def run_all():
    """Run both API and UI (for development)."""
    import threading

    # Run API in a thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    print("🚀 API server started on http://localhost:8000")
    print("🚀 Starting Streamlit UI...")

    # Run UI in main thread
    run_ui()


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Social Media AI Agent - Generate and publish posts with human approval"
    )

    parser.add_argument(
        "command",
        choices=["api", "ui", "mcp", "all"],
        default="all",
        nargs="?",
        help="Command to run: api (FastAPI), ui (Streamlit), mcp (MCP server), all (API + UI)",
    )

    args = parser.parse_args()

    print("🤖 Social Media AI Agent")
    print("=" * 40)

    if args.command == "api":
        print("Starting FastAPI server...")
        run_api()
    elif args.command == "ui":
        print("Starting Streamlit UI...")
        run_ui()
    elif args.command == "mcp":
        print("Starting MCP server...")
        run_mcp()
    else:
        print("Starting both API and UI...")
        run_all()


if __name__ == "__main__":
    main()
