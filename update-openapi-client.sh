#!/bin/bash

# Script to regenerate the OpenAPI client for cyborgdb-py
# Run from project root: ./update-openapi-client.sh

set -e  # Exit on any error

echo "🔄 Updating OpenAPI Client..."

# Check if openapi.json exists
if [ ! -f "openapi.json" ]; then
    echo "❌ Error: openapi.json not found in current directory"
    echo "Please make sure you're running this from the project root"
    exit 1
fi

# Check if openapi-generator is installed
if ! command -v openapi-generator &> /dev/null; then
    echo "❌ Error: openapi-generator not found"
    echo "Please install it with: brew install openapi-generator"
    exit 1
fi

# Generate the client (will overwrite existing files)
echo "🔧 Generating client..."
openapi-generator generate \
    -i openapi.json \
    -g python \
    -o . \
    --package-name cyborgdb.openapi_client \
    --additional-properties=generateSourceCodeOnly=true

echo "✅ OpenAPI client updated successfully!"