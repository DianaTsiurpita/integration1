"""Evaluate a Ridge model on held-out data and save a production model."""
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from .model import CAT, NUM, TARGET, load_training_data, train_model

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

def main() -> None:
    df = load_training_data()
    X, y = df[NUM + CAT], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # All preprocessing is fitted only on training data (no leakage).
    model = train_model(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    print(f"[train] Train: R2={r2_score(y_train, train_pred):.4f}, MAE={mean_absolute_error(y_train, train_pred):.0f} USD")
    print(f"[train] Test:  R2={r2_score(y_test, test_pred):.4f}, MAE={mean_absolute_error(y_test, test_pred):.0f} USD")
    print(f"[train] Split: train={len(X_train)}, test={len(X_test)}, random_state=42")
    # Refit on all observations after unbiased holdout evaluation for deployment.
    final_model = train_model(X, y)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACTS / "model.pkl"
    joblib.dump(final_model, artifact_path)
    print(f"[train] artifact persisted: {artifact_path}")

if __name__ == "__main__":
    main()
