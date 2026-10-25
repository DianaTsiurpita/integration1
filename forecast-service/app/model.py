"""Definition of the Ridge model and preprocessing pipeline for Ames Housing."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ames_train.csv"
NUM = ["GrLivArea", "TotalBsmtSF", "GarageArea", "YearBuilt", "OverallQual"]
CAT = ["MSZoning"]
TARGET = "SalePrice"
CSV_COLUMNS = {
    "GrLivArea": "GrLivArea",
    "TotalBsmtSF": "TotalBsmtSF",
    "GarageArea": "GarageArea",
    "YearBuilt": "YearBuilt",
    "OverallQual": "OverallQual",
    "MSZoning": "MSZoning",
    "SalePrice": "SalePrice",
}

def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    # Accept both the original Ames headers and already-normalized headers.
    df = df.rename(columns=CSV_COLUMNS)
    required = NUM + CAT + [TARGET]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Training dataset is missing required columns: {sorted(missing)}")
    return df[required].dropna(subset=[TARGET])

def train_model(X=None, y=None) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUM),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])
    pipe = Pipeline([("preprocess", preprocessor), ("regressor", Ridge(alpha=1.0))])
    if X is None or y is None:
        df = load_training_data()
        X, y = df[NUM + CAT], df[TARGET]
    pipe.fit(X, y)
    return pipe
