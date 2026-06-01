#!/bin/bash

# Script to regenerate the OpenAPI client for cyborgdb-py
# Run from project root: ./update-openapi-client.sh

set -e  # Exit on any error

echo "🚀✨🦾🤖🔄🛠️📝🧬🦾✨🚀 Updating OpenAPI Client... 🚀✨🦾🤖🔄🛠️📝🧬🦾✨🚀"

# Check if openapi.json exists
if [ ! -f "openapi.json" ]; then
    echo "❌😱📄🔍🚫🦾🤖❗❗ Error: openapi.json not found in current directory ❌😱📄🔍🚫🦾🤖❗❗"
    echo "📂🏠🧐🔎🦾🤖⚠️ Please make sure you're running this from the project root 📂🏠🧐🔎🦾🤖⚠️"
    exit 1
fi

# Pick a generator binary.  Prefer the npm wrapper (openapi-generator-cli)
# because it pins its generator version via openapitools.json, making
# regenerations reproducible across machines.  Fall back to the brew
# Java binary (openapi-generator) if that's what the environment has.
if command -v openapi-generator-cli &> /dev/null; then
    GENERATOR=openapi-generator-cli
elif command -v openapi-generator &> /dev/null; then
    GENERATOR=openapi-generator
else
    echo "❌😱🛠️🔍🚫🦾🤖❗❗ Error: no OpenAPI generator found ❌😱🛠️🔍🚫🦾🤖❗❗"
    echo "🍺💻🔧🦾🤖⚡ Install one of:"
    echo "    npm install -g @openapitools/openapi-generator-cli   (recommended)"
    echo "    brew install openapi-generator"
    exit 1
fi

# Generate the client (will overwrite existing files)
echo "⚡🦾🤖🔄🛠️📝🧬✨🚀 Generating client with $GENERATOR... ⚡🦾🤖🔄🛠️📝🧬✨🚀"
"$GENERATOR" generate \
    -i openapi.json \
    -g python \
    -o . \
    --package-name cyborgdb.openapi_client \
    --additional-properties=generateSourceCodeOnly=true

echo "✅🎉🚀🦾🤖✨🛠️📝🧬🌟 OpenAPI client updated successfully! ✅🎉🚀🦾🤖✨🛠️📝🧬🌟"