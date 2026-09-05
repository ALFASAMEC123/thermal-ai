#!/bin/bash
# Deploy to GitHub and trigger portable bundle build
# Run this from Linux Mint: ./deploy_to_github.sh

set -e

echo "=== Thermal AI - Deploy to GitHub Actions ==="
echo

# Check if git repo exists
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: thermal AI portable bundle"
fi

# Get GitHub repo URL
if [ -z "$1" ]; then
    echo "Usage: ./deploy_to_github.sh <github-repo-url>"
    echo "Example: ./deploy_to_github.sh https://github.com/username/thermal-ai.git"
    exit 1
fi

REPO_URL=$1
BRANCH=${2:-main}

echo "Remote: $REPO_URL"
echo "Branch: $BRANCH"
echo

# Add remote if not exists
if ! git remote | grep -q origin; then
    git remote add origin "$REPO_URL"
else
    git remote set-url origin "$REPO_URL"
fi

# Push
echo "Pushing to GitHub..."
git push -u origin "$BRANCH"

echo
echo "=== Deployed! ==="
echo
echo "Next steps:"
echo "1. Go to: https://github.com/$(basename "$REPO_URL" .git)/actions"
echo "2. Click 'Build Portable Windows Bundle' → 'Run workflow' → 'Run workflow'"
echo "3. Wait 15-25 minutes"
echo "4. Download ZIP from: Actions → build run → Artifacts"
echo
echo "OR create a release tag (auto-builds + creates Release):"
echo "  git tag v1.0.0"
echo "  git push origin v1.0.0"
echo "  Then check: https://github.com/$(basename "$REPO_URL" .git)/releases"