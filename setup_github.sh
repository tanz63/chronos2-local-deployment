#!/bin/bash
# GitHub 仓库设置脚本

# 由于 gh CLI 需要交互式认证，使用 git 命令直接推送

REPO_NAME="chronos2-local-deployment"
GITHUB_USER="tanzheng"  # 用户需要替换为自己的用户名

echo "Setting up GitHub repository..."
echo "Repository: $REPO_NAME"

# 检查远程仓库是否存在
if git remote | grep -q origin; then
    echo "Remote 'origin' already exists"
else
    # 添加远程仓库
    git remote add origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"
    echo "Remote added: https://github.com/$GITHUB_USER/$REPO_NAME.git"
fi

# 推送代码
echo "Pushing code to GitHub..."
git push -u origin main

echo "Done! Repository URL: https://github.com/$GITHUB_USER/$REPO_NAME"
