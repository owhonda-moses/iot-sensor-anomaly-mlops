#!/usr/bin/env python3
"""
Setup script for MLflow and Prefect local development environment
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories"""
    dirs = [
        "iot_models/baseline",
        "iot_models/optimized", 
        "mlruns",
        "logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def create_mlflow_config():
    """Create MLflow configuration files"""
    
    # Create .mlflowrc file for default settings
    mlflow_config = """
[mlflow]
default_artifact_root = ./mlruns
backend_store_uri = sqlite:///mlflow.db
"""
    
    with open(".mlflowrc", "w") as f:
        f.write(mlflow_config)
    print("✅ Created .mlflowrc configuration file")

def create_prefect_config():
    """Create Prefect configuration"""
    
    prefect_config = """
# Prefect Configuration
# This file configures Prefect for local development

# To use this configuration:
# 1. Start Prefect server: prefect server start
# 2. In another terminal: prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
# 3. Run your flows!

PREFECT_API_URL=http://127.0.0.1:4200/api
PREFECT_LOGGING_LEVEL=INFO
"""
    
    with open("prefect.env", "w") as f:
        f.write(prefect_config)
    print("✅ Created prefect.