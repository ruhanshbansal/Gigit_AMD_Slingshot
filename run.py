"""
Gigit — Quick Launcher
======================
Double-click this file or run `python run.py` to start the dashboard.
"""

import os
import subprocess
import sys

if __name__ == "__main__":
    # Always run from the project root, regardless of where this script is called from
    project_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard = os.path.join(project_dir, "frontend", "underwriter_dashboard.py")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run", dashboard,
    ])
