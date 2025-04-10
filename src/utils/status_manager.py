import json
from datetime import datetime
from src.utils.path_manager import BASE_DIR

STATUS_FILE = BASE_DIR / "project_status.json"

def update_status(key, value=None):
    """Update project status with a timestamp or custom value."""
    status = {}
    if STATUS_FILE.exists():
        with open(STATUS_FILE, 'r') as f:
            status = json.load(f)

    status[key] = value if value else datetime.now().isoformat()

    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=4)


def get_status_summary():
    if not STATUS_FILE.exists():
        print("No project_status.json found.")
        return

    with open(STATUS_FILE, 'r') as f:
        status = json.load(f)

    print("\n📊 Project Status Summary:\n----------------------------")
    for key, value in status.items():
        print(f"{key}: {value}")
    print("----------------------------\n")
