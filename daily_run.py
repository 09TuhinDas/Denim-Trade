import os
import datetime 
import subprocess

# === Config ===
RETRAIN_MODELS = False  # Set to False to skip retraining
USE_STACKED_MODEL = True
CONFIDENCE_THRESHOLD = 0.85

# === Paths ===
DATE = datetime.datetime.now().strftime("%Y-%m-%d")
now = datetime.datetime.now()
TIMESTAMP = now.strftime("%Y%m%d_%H%M%S")
SCREENER_LOG = f"logs/screener_runs/screener_stacked_{TIMESTAMP}.csv"

def run_command(description, command):
    print(f"\n🔄 {description}...")
    result = subprocess.run(command, shell=True)
    if result.returncode == 0:
        print(f"✅ {description} completed.")
    else:
        print(f"❌ {description} failed. Please check logs.")

def main():
    print(f"\n🗓️ Starting Daily Routine – {DATE}")

    # 1. Update Macro Data
    run_command("Fetching macro data", "python -m src.utils.macro_features")

    # 2. Run Regime Detection
    run_command("Running regime detection", "python -m src.ml.regime_detector")

    # 3. Optional: Retrain models
    if RETRAIN_MODELS:
        run_command("Retraining models", "python -m src.ml.retrain_batches")

    # 4. Run Screener
    mode = "stacked" if USE_STACKED_MODEL else "xgb"
    screener_cmd = (
        f"python -m src.screener.screen_market "
        f"--mode {mode} "
        f"--confidence {CONFIDENCE_THRESHOLD} "
        f"--save_path {SCREENER_LOG}"
    )
    run_command(f"Running screener ({mode})", screener_cmd)

    print(f"\n✅ Daily routine completed. Screener output saved to: {SCREENER_LOG}")

if __name__ == "__main__":
    main()
