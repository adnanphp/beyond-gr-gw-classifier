"""Physics-based train / validation / test split."""
import numpy as np
import pandas as pd


def physics_based_split(df, cfg, seed=42):
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["mass_ratio"] = df["mass1"] / df["mass2"]

    test_type = cfg["deviation"]["test_type"]
    train_types = cfg["deviation"]["train_types"]

    # GR + types A, B — model has seen these
    known = df[df["type"].isin(["GR"] + train_types)].copy()
    # Type C — model has NOT seen this
    unseen_c = df[df["type"] == test_type].copy()

    # Split known into train/val/test_seen
    idx = rng.permutation(len(known))
    known = known.iloc[idx].reset_index(drop=True)
    n = len(known)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train_df = known.iloc[:n_train].reset_index(drop=True)
    val_df = known.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_seen_df = known.iloc[n_train + n_val:].reset_index(drop=True)

    # For test_unseen: 50% GR + 50% type C
    n_gr = len(df[df["type"] == "GR"])
    # Take GR samples not already in test_seen
    gr_pool = df[df["type"] == "GR"].sample(frac=1.0, random_state=seed)
    n_c = len(unseen_c)
    n_gr_in_unseen = min(n_c, n_gr)  # match C count

    gr_for_unseen = gr_pool.iloc[:n_gr_in_unseen].reset_index(drop=True)
    c_for_unseen = unseen_c.iloc[:n_gr_in_unseen].reset_index(drop=True)

    test_unseen_df = pd.concat([gr_for_unseen, c_for_unseen], ignore_index=True)
    test_unseen_df = test_unseen_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_seen_df, test_unseen_df


def save_splits(train_df, val_df, test_seen_df, test_unseen_df,
                meta_dir="data/metadata"):
    from pathlib import Path
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(meta_dir / "train.csv", index=False)
    val_df.to_csv(meta_dir / "val.csv", index=False)
    test_seen_df.to_csv(meta_dir / "test_seen.csv", index=False)
    test_unseen_df.to_csv(meta_dir / "test_unseen.csv", index=False)

    print(f"train:       {len(train_df)}  ({train_df['label'].value_counts().to_dict()})")
    print(f"val:         {len(val_df)}  ({val_df['label'].value_counts().to_dict()})")
    print(f"test_seen:   {len(test_seen_df)}  ({test_seen_df['label'].value_counts().to_dict()})")
    print(f"test_unseen: {len(test_unseen_df)}  ({test_unseen_df['label'].value_counts().to_dict()})")
    print(f"  test_unseen types: {test_unseen_df['type'].value_counts().to_dict()}")
