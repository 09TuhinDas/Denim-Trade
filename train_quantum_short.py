from src.ml import quantum_short_model

if __name__ == "__main__":
    print("📦 Training short model using patched logic...")
    X, y = quantum_short_model.load_short_batches()
    print("🔍 Global label counts before training:", y.value_counts().to_dict())

    quantum_short_model.train_short_model(X, y)
    
