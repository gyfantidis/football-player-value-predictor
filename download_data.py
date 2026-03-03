"""
Download raw data files from Google Drive into data/raw/.

Usage:
    pip install gdown
    python download_data.py
"""

import subprocess
import sys
from pathlib import Path

FOLDER_URL = "https://drive.google.com/drive/folders/1c8XFIAOFcBhUafRwAoAZJvzFjdzmKrzT"
OUTPUT_DIR = Path("data/raw")

EXPECTED_FILES = [
    "players.csv",
    "appearances.csv",
    "player_valuations.csv",
    "club_games.csv",
    "game_lineups.csv",
    "clubs.csv",
    "transfers.csv",
    "game_events.csv",
    "games.csv",
    "competitions.csv",
]


def install_gdown():
    print("Installing gdown...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q", "--break-system-packages"])


def main():
    # Ensure gdown is available
    try:
        import gdown
    except ImportError:
        install_gdown()
        import gdown

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading data from Google Drive into {OUTPUT_DIR}/")
    print("This may take a few minutes for large files (appearances.csv ~500MB).\n")

    gdown.download_folder(
        url=FOLDER_URL,
        output=str(OUTPUT_DIR),
        quiet=False,
        use_cookies=False,
    )

    # Verify
    print("\nVerifying downloaded files:")
    missing = []
    for fname in EXPECTED_FILES:
        path = OUTPUT_DIR / fname
        if path.exists():
            size_mb = path.stat().st_size / 1_000_000
            print(f"  ✓  {fname} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗  {fname} — NOT FOUND")
            missing.append(fname)

    if missing:
        print(f"\nWarning: {len(missing)} file(s) missing. Check the Drive folder is publicly shared.")
    else:
        print("\nAll files downloaded. You can now run the notebooks in order (01 → 10).")


if __name__ == "__main__":
    main()
