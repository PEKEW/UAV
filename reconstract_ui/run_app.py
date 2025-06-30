"""
Entry point to run the refactored UAV Battery Health Monitoring Application
"""
import sys
import os

# Add the parent directory to the path to import reconstract_ui
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reconstract_ui.app import main

if __name__ == "__main__":
    main()