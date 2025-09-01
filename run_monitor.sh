#!/bin/bash

# Navigate to the project directory to ensure all relative paths work correctly
cd /home/giuseppe/xrp-regime-detection/

# Activate the virtual environment and run the monitor script,
# logging all output to the monitor.log file.
/home/giuseppe/xrp-regime-detection/.venv/bin/python monitor_and_optimize.py >> /home/giuseppe/xrp-regime-detection/data/monitor.log 2>&1