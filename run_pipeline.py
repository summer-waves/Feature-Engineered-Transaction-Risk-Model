from pathlib import Path
from src.data_prep import load_transactions
from src.features import build_origin_account_features
from src.model import add_risk_score

def main():
    raw_path = Path("data/raw") / "transactions.csv"
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load raw transactions
    df = load_transactions(raw_path, n_rows=500000)

    # 2) Build origin account features
    feat = build_origin_account_features(df)
    feat.to_csv(processed_dir / "origin_account_features.csv")

    # 3) Add risk score
    feat_with_score = add_risk_score(feat)
    feat_with_score.to_csv(processed_dir / "origin_account_features_scored.csv")

    # 4) Print top 20 high-risk accounts
    print(feat_with_score.sort_values("risk_score", ascending=False).head(20))

if __name__ == "__main__":
    main()