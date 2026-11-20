"""Supported Streamlit entrypoint; dashboard.py uses the same application."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('dashboard.py')), run_name='__main__')
