# internal-tool

Short internal tool for extracting data from DXF/PDF/OCR sources and serving a FastAPI API for testing and development.

## Requirements
- Python 3.11+ (Windows tested)
- `requirements.txt` (use the project venv)

## Setup
1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the app

Start the FastAPI server (use `python -m uvicorn` to avoid broken console scripts):

```powershell
& .\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 in your browser.

## Tests

Run the test suite:

```powershell
& .\venv\Scripts\Activate.ps1
pytest -q
```

## Git push (HTTPS) — common auth issue on Windows

If `git push` fails with `Permission denied` or a 403 referencing another account, remove the cached credential in Windows Credential Manager and use a Personal Access Token (PAT):

1. Open Credential Manager → Windows Credentials → remove any `git:` or `github.com` entries for the wrong user.
2. Create a PAT in GitHub (Settings → Developer settings → Personal access tokens) with `repo` scope.
3. Push and enter your GitHub username and the PAT as the password:

```powershell
git push -u origin main
# Username: <your-username>
# Password: <your-PAT>
```

## Git push (SSH) — alternative (one-time setup)

```powershell
# generate a key
ssh-keygen -t ed25519 -C "your-email@example.com"
# copy the public key to GitHub Settings → SSH and GPG keys
git remote set-url origin git@github.com:your-username/internal-tool.git
git push -u origin main
```

## Notes
- If you hit compiled-extension errors (NumPy / OpenCV), try installing a matching wheel as done in this project (e.g. `opencv-python-headless` compatible with the installed `numpy`).
