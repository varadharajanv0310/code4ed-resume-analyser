"""
Main entry point for Resume-JD Analyzer
This file is required for Streamlit Cloud deployment
"""

import streamlit as st
import sys
import os
from pathlib import Path
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Add the app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

# Import and run the main application
try:
    from main import main

    # Set page config
    st.set_page_config(
        page_title="Resume-JD Analyzer Pro",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Run the enhanced main application
    main()

except ImportError as e:
    st.error(f"Import error: {e}")
    st.info(
        "If this is your first deployment, some dependencies might still be installing. Please refresh in a few minutes.")

except Exception as e:
    st.error(f"Application error: {e}")

    st.info("Please check the logs for more details.")

