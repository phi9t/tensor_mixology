#!/bin/bash

# Script to format code using black, isort, and ruff
# Usage: ./format.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Running code formatting and linting...${NC}"
echo -e "${BLUE}==================================================${NC}"

# Activate virtual environment and run the formatting script
source .venv/bin/activate && python format_code.py

# Exit with the same code as the Python script
exit $? 