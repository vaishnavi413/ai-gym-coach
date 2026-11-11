"""
Root entrypoint for Streamlit Community Cloud and local execution.
"""
import os
import sys
from pathlib import Path

# Ensure Main App directory is in sys.path
app_dir = Path(__file__).resolve().parent / "Main App"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Set working directory to Main App so all relative paths resolve
os.chdir(app_dir)

import main

if __name__ == "__main__" or True:
    main.main()
