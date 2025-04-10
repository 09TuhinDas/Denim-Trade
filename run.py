import sys
from pathlib import Path

# Add ./src to PYTHONPATH at runtime
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from main import main

if __name__ == "__main__":
    main()
