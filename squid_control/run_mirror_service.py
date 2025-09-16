#!/usr/bin/env python3
"""
Simple runner script for the mirror service.

This script provides backward compatibility for users who want to run
the mirror service directly without using the new module structure.

Usage:
    python run_mirror_service.py --cloud-service-id "mirror-microscope-control-squid-2" --local-service-id "microscope-control-squid-2"
"""

import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mirror.cli import main

if __name__ == "__main__":
    main()
