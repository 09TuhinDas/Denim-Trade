import os
import subprocess
from datetime import datetime
from drift_monitor import compute_drift_scores, load_combined_training_data, load_live_data

DRIFT_THRESHOLD = 0.3
RETRAIN_COMMAND = "python retrain_xgboost.py"
RETRAIN_LOG = "logs/retrain_log.txt"


def should_retrain():
    print("\n📊 Checking for retraining trigger...")
    train_df = load_combined_training_data()
    live_df = load_live_data()
    drift_scores = compute_drift_scores(train_df, live_df)

    max_drift = max([
        float(score) for score in drift_scores.values()
        if isinstance(score, float)
    ], default=0.0)

    print(f"Max drift detected: {max_drift:.4f}")
    return max_drift > DRIFT_THRESHOLD


def trigger_retrain():
    print("\n🔁 Drift threshold exceeded. Triggering retraining...")
    os.makedirs("logs", exist_ok=True)
    with open(RETRAIN_LOG, "a") as f:
        f.write(f"[{datetime.now()}] Retraining triggered due to drift.\n")
    subprocess.run(RETRAIN_COMMAND, shell=True)


def main():
    if should_retrain():
        trigger_retrain()
    else:
        print("✅ No retraining needed. Drift within acceptable range.")


if __name__ == "__main__":
    main()
