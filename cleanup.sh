#!/bin/bash

echo "Starting selective cleanup of unnecessary OpenAPI generated files..."

# Delete OpenAPI generator metadata but keep the generated Python client code
echo "Removing OpenAPI generator metadata files..."
find . -path "*/.openapi-generator/VERSION" -type f -delete
find . -path "*/.openapi-generator/FILES" -type f -delete
find . -name "openapitools.json" -type f -delete
find . -name ".openapi-generator-ignore" -type f -delete

# Delete test files but keep essential ones
echo "Removing generated test files..."
# Keep the test files in your primary working directory
find . -path "*/test/test_*.py" ! -path "./tests/*" -type f -delete

# Delete generated documentation (keeping essential README files)
echo "Removing generated documentation..."
find . -path "*/docs/*.md" ! -name "README.md" -type f -delete

# Delete CI files
echo "Removing CI/CD configuration files..."
find . -name ".travis.yml" -type f -delete
find . -name ".gitlab-ci.yml" -type f -delete
find . -name "git_push.sh" -type f -delete
# Keep tox.ini as it might contain essential configuration
rm -rf .github/workflows 2>/dev/null || true

# Delete cache files
echo "Removing Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} \; 2>/dev/null || true
find . -name "*.pyc" -type f -delete
find . -name ".pytest_cache" -type d -exec rm -rf {} \; 2>/dev/null || true

# Delete egg-info directory
echo "Removing Python egg-info directories..."
find . -name "*.egg-info" -type d -exec rm -rf {} \; 2>/dev/null || true

echo "Cleanup complete!"