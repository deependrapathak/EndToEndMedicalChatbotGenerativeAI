# EndToEndMedicalChatbotGenerativeAI

# How to run?
### STEPS:
```bash
Project repo: https://github.com/deependrapathak/EndToEndMedicalChatbotGenerativeAI
```

### Create Conda environment
```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

### Trubleshooting large file uploads to github
## Increase git http buffer
## Find large file
## Reset if not
```bash
git config --global http.postBuffer 524288000
git rm -r --cached .venv venv data
git commit -m "Remove large files and add gitignore"
git push
git rev-list --objects --all | \
git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
grep blob | sort -k3 -n | tail -10

git reset --soft HEAD~1
git add .
git commit -m "Clean commit"
git push --force

```