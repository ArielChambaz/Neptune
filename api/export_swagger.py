#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Swagger Schema
Script to export the current API OpenAPI schema to a JSON file.
"""

import json
import argparse
from pathlib import Path
import sys

# Add api directory to path to allow importing main
sys.path.append(str(Path(__file__).parent))

from main import app

def export_swagger(output_path: str = "openapi.json"):
    """
    Export the OpenAPI schema to a file.
    """
    print(f"Generating OpenAPI schema...")

    # Get the schema
    openapi_schema = app.openapi()

    # Write to file
    with open(output_path, "w") as f:
        json.dump(openapi_schema, f, indent=2)

    print(f"✅ OpenAPI schema exported to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FastAPI OpenAPI schema")
    parser.add_argument("--output", "-o", default="openapi.json", help="Output JSON file path")
    args = parser.parse_args()

    export_swagger(args.output)
