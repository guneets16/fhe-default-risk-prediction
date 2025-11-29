#!/bin/bash

# 🚀 Push to GitHub - Interactive Setup Script
# This script will guide you through pushing your code to a new GitHub repository

echo "=================================================================="
echo "  🚀 GitHub Push Setup for FHE Credit Risk Prediction"
echo "=================================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -f "train_credit_model.py" ]; then
    echo "❌ Error: Please run this script from the fhe-default-risk-prediction directory"
    exit 1
fi

# Step 1: Get GitHub repository URL from user
echo "📋 STEP 1: GitHub Repository URL"
echo "=================================================================="
echo ""
echo "First, create a new repository on GitHub:"
echo "  1. Go to: https://github.com/new"
echo "  2. Repository name: fhe-default-risk-prediction"
echo "  3. DO NOT check: Add README, .gitignore, or license"
echo "  4. Click 'Create repository'"
echo "  5. Copy the repository URL"
echo ""
read -p "Enter your GitHub repository URL: " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "❌ No URL provided. Exiting."
    exit 1
fi

echo "✅ Using repository: $REPO_URL"
echo ""

# Step 2: Initialize Git
echo "📋 STEP 2: Initialize Git"
echo "=================================================================="
echo ""

if [ -d ".git" ]; then
    echo "⚠️  Git is already initialized in this directory"
    read -p "Do you want to continue? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        echo "Exiting."
        exit 0
    fi
else
    echo "Initializing Git repository..."
    git init
    git branch -M main
    echo "✅ Git initialized on 'main' branch"
fi

echo ""

# Step 3: Check files to be committed
echo "📋 STEP 3: Review Files"
echo "=================================================================="
echo ""
echo "Files that will be committed:"
git add .
git status --short

echo ""
read -p "Does this look correct? (y/n): " FILES_OK

if [ "$FILES_OK" != "y" ]; then
    echo "Please review your .gitignore file and adjust as needed."
    echo "Then run this script again."
    exit 0
fi

# Step 4: Create commit
echo ""
echo "📋 STEP 4: Create Initial Commit"
echo "=================================================================="
echo ""
echo "Creating commit..."

git commit -m "Initial commit: Privacy-Preserving Credit Default Risk Prediction with FHE

- Complete ML pipeline for credit default prediction using UCI dataset
- FastAPI backend with TenSEAL FHE encryption support (CKKS scheme)
- Interactive Streamlit frontend with client-side encryption
- Three trained models: Logistic Regression, Random Forest, Gradient Boosting
- Comprehensive documentation (README, HOW_IT_WORKS, API docs)
- Automated setup scripts for quick deployment
- MIT License"

echo "✅ Commit created successfully"
echo ""

# Step 5: Add remote
echo "📋 STEP 5: Connect to GitHub"
echo "=================================================================="
echo ""

# Check if remote already exists
if git remote | grep -q "origin"; then
    echo "⚠️  Remote 'origin' already exists"
    echo "Current remote: $(git remote get-url origin)"
    read -p "Do you want to replace it? (y/n): " REPLACE_REMOTE
    
    if [ "$REPLACE_REMOTE" = "y" ]; then
        git remote remove origin
        git remote add origin "$REPO_URL"
        echo "✅ Remote updated"
    fi
else
    git remote add origin "$REPO_URL"
    echo "✅ Remote added: $REPO_URL"
fi

echo ""

# Step 6: Push to GitHub
echo "📋 STEP 6: Push to GitHub"
echo "=================================================================="
echo ""
echo "⚠️  IMPORTANT: You may be prompted for GitHub credentials"
echo ""
echo "If using HTTPS (username/password):"
echo "  - Username: Your GitHub username"
echo "  - Password: Use a Personal Access Token (NOT your password)"
echo "    Generate at: https://github.com/settings/tokens"
echo ""
echo "If using SSH: Make sure your SSH key is added to GitHub"
echo ""
read -p "Ready to push? Press Enter to continue..."

echo ""
echo "Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================================="
    echo "  🎉 SUCCESS! Your code is now on GitHub!"
    echo "=================================================================="
    echo ""
    echo "Your repository: $(echo $REPO_URL | sed 's/\.git$//')"
    echo ""
    echo "Next steps:"
    echo "  1. Visit your repository on GitHub"
    echo "  2. Add topics/tags (fhe, machine-learning, privacy, etc.)"
    echo "  3. Verify README displays correctly"
    echo "  4. Share with the community!"
    echo ""
else
    echo ""
    echo "=================================================================="
    echo "  ❌ Push Failed"
    echo "=================================================================="
    echo ""
    echo "Common issues:"
    echo ""
    echo "1. Authentication failed:"
    echo "   - Generate Personal Access Token: https://github.com/settings/tokens"
    echo "   - Use token as password (not your GitHub password)"
    echo ""
    echo "2. Remote rejected:"
    echo "   - Make sure repository exists on GitHub"
    echo "   - Check repository URL is correct"
    echo ""
    echo "3. Connection issues:"
    echo "   - Check internet connection"
    echo "   - Try SSH instead of HTTPS"
    echo ""
    echo "For detailed troubleshooting, see: GIT_SETUP_GUIDE.md"
    echo ""
    exit 1
fi

