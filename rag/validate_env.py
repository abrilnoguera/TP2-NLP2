"""Utility to validate required environment variables for the rag app.

This script prints which environment variables are set and issues instructions.
It intentionally does NOT print the actual values of the keys to avoid leaking secrets.
"""
import os
import pathlib
import re

REQUIRED = ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "GROQ_API_KEY"]

def main():
    # Try to load a .env file if present in repo root or current directory
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    candidate_envs = [repo_root / '.env', pathlib.Path(__file__).resolve().parent / '.env']
    for candidate in candidate_envs:
        if candidate.exists():
            load_env_file(candidate)
            print(f"🔐 Loaded environment variables from {candidate}")
            break
    ok = True
    for k in REQUIRED:
        if os.getenv(k):
            print(f"✅ {k} is defined")
        else:
            print(f"❌ {k} is NOT defined")
            ok = False
    if os.getenv("PINECODE_API_KEY"):
        print("⚠️ Detected PINECODE_API_KEY: it looks like there might be a typo - rename to PINECONE_API_KEY")
    if not ok:
        print("\nPlease copy .env.example to .env and fill your keys, or `export` them in your shell session.")

if __name__ == '__main__':
    # Load_env_file must be defined before running main; the main call is at the end of the file
    pass


def load_env_file(path: pathlib.Path):
    """Minimal .env file loader to populate os.environ for validation.

    This will parse lines of the form `KEY=VALUE`, ignore comments and blank lines,
    and set them in `os.environ` if not already present.
    """
    pattern = re.compile(r"^\s*(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*)\s*$")
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = pattern.match(line)
            if not m:
                continue
            key = m.group('key')
            # Remove surrounding quotes from the value if present
            value = m.group('value')
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            # Only set if not already in env to avoid overriding user settings
            if not os.getenv(key):
                os.environ[key] = value
    return


if __name__ == '__main__':
    main()
