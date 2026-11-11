"""
Root runner for AI Gym Coach Streamlit App.
Allows running `streamlit run main.py` directly from the repository root.
"""
import os
import sys
from pathlib import Path

# Add Main App directory to sys.path
app_dir = Path(__file__).resolve().parent / "Main App"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Set working directory to Main App
os.chdir(app_dir)

import main

if __name__ == "__main__":
    main.main()
