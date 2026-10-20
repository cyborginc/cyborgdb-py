#!/bin/bash

# Script to regenerate the OpenAPI client for cyborgdb-py
# Run from project root: ./update-openapi-client.sh [path/to/openapi.json]

set -e  # Exit on any error

# The spec's source of truth is cyborgdb-tests; pass a path to override.
SPEC="${1:-../cyborgdb-tests/openapi.json}"

echo "🚀✨🦾🤖🔄🛠️📝🧬🦾✨🚀 Updating OpenAPI Client from $SPEC... 🚀✨🦾🤖🔄🛠️📝🧬🦾✨🚀"

if [ ! -f "$SPEC" ]; then
    echo "❌😱📄🔍🚫🦾🤖❗❗ Error: $SPEC not found ❌😱📄🔍🚫🦾🤖❗❗"
    echo "📂🏠🧐🔎🦾🤖⚠️ Usage: $0 [path/to/openapi.json], run from the project root 📂🏠🧐🔎🦾🤖⚠️"
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
    -i "$SPEC" \
    -g python \
    -o . \
    --package-name cyborgdb.openapi_client \
    --additional-properties=generateSourceCodeOnly=true

# Async client: same API surface, httpx transport. Lives in its own package
# (models included) so neither tree needs hand edits to share anything.
echo "⚡🦾🤖🔄🛠️📝🧬✨🚀 Generating async (httpx) client with $GENERATOR... ⚡🦾🤖🔄🛠️📝🧬✨🚀"
"$GENERATOR" generate \
    -i "$SPEC" \
    -g python \
    -o . \
    --package-name cyborgdb.openapi_client_async \
    --additional-properties=generateSourceCodeOnly=true,library=httpx

echo "✅🎉🚀🦾🤖✨🛠️📝🧬🌟 OpenAPI client updated successfully! ✅🎉🚀🦾🤖✨🛠️📝🧬🌟"