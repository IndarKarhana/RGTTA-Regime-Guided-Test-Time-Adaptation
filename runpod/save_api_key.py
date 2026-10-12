"""Run locally in an interactive terminal; never paste the key into chat."""
import getpass
import os
from pathlib import Path


def save_key(path):
    key = getpass.getpass("RunPod API key (hidden): ").strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError("Expected a nonempty API key without whitespace")
    # Exclusive creation avoids overwriting files or following an existing symlink.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(key)
    print("Key saved locally with owner-only permissions. Return to chat and say ready.")


if __name__ == "__main__":
    save_key(Path("/private/tmp/rgtta-runpod-api-key"))
