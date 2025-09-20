"""
Application runner for Resume Relevance Check System
"""
import streamlit.web.cli as stcli
import sys
import os
from pathlib import Path


def main():
    """Run the Streamlit application"""

    # Get the path to the main app file
    app_path = Path(__file__).parent / "app" / "main.py"

    # Set up sys.argv for Streamlit
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port=8501",
        "--server.address=localhost",
        "--server.headless=false"
    ]

    # Run Streamlit
    stcli.main()


if __name__ == "__main__":
    main()