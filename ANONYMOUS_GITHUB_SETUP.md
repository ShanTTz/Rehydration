# Anonymous GitHub Setup

This export contains no Git history, remotes, or contributor metadata. The
source workspace remains separate.

Before the first push, use the identity and account required by the anonymous
repository service. The commands below use a non-routable placeholder identity
for the initial commit; they do not anonymize a GitHub account or hosting logs.

```bash
git lfs install
git init -b main
git config user.name "Anonymous"
git config user.email "anonymous@example.invalid"
git add .
git status --short
git commit -m "Anonymous reproduction release"
git remote add origin <anonymous-repository-url>
git push -u origin main
```

Check the staged file list before committing. Do not copy a `.git` directory
from another workspace, add API credentials, or add raw external-platform
downloads or participant-level data.
