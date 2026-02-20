#!/usr/bin/env python
"""Setup git repository and push to GitHub"""

import os
from git import Repo
from git.exc import InvalidGitRepositoryError

os.chdir(r"c:\Users\khit\Downloads\Major")

try:
    # Initialize repo
    repo = Repo.init(".")
    print("✓ Git repository initialized")
    
    # Configure user for commits
    with repo.config_writer() as config:
        config.set_value("user", "name", "khit").release()
        config.set_value("user", "email", "user@example.com").release()
    
    # Stage all files
    repo.index.add([item for item in os.listdir(".") if item != ".git"])
    print("✓ Files staged")
    
    # Create initial commit
    repo.index.commit("first commit")
    print("✓ Initial commit created")
    
    # Create/move to main branch
    if repo.active_branch.name != "main":
        # Rename master to main
        repo.heads.master.rename("main")
        print("✓ Branch renamed to main")
    else:
        print("✓ Already on main branch")
    
    # Add remote origin
    try:
        repo.delete_remote("origin")
    except:
        pass
    
    origin = repo.create_remote(
        "origin",
        "https://github.com/238x1a05h8-hash/A-machine-learning-approach-to-flood-prediction.git"
    )
    print("✓ Remote origin added")
    
    # Push to GitHub
    origin.push(repo.active_branch, set_upstream=True)
    print("✓ Pushed to GitHub")
    
    print("\n✅ All git operations completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
