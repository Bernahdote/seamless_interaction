# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from seamless_interaction.fs import DatasetConfig, SeamlessInteractionFS


def download_1gb_sample_archive():
    """
    Download ~1GB of samples using selective archives.

    Traditional archive-based approach for quick exploration on laptops.
    """
    config = DatasetConfig(label="improvised", split="dev", num_workers=4)
    fs = SeamlessInteractionFS(config=config)

    # Download specific archives (~1GB total)
    fs.download_batch_from_hf(batch_idx=0, archive_list=[0])
    print("✅ Downloaded ~1GB sample from HF (archive-based)")


def download_single_batch():
    """
    Download a complete batch (~50-100GB).

    Good for substantial local exploration and development.
    """
    config = DatasetConfig(label="improvised", split="dev", num_workers=8)
    fs = SeamlessInteractionFS(config=config)

    # Download complete batch
    fs.download_batch_from_hf(batch_idx=0)
    print("✅ Downloaded single batch (~50-100GB)")


def download_multiple_batches():
    """
    Download multiple batches for training datasets.

    Suitable for model training and large-scale analysis.
    """
    config = DatasetConfig(label="improvised", split="train", num_workers=8)
    fs = SeamlessInteractionFS(config=config)

    # Download first 3 batches of training data (~150GB+)
    # Updated: Download 15 batches --> 750GB+ of data and thus 100hours+ of speech  

    for batch_idx in range(15):
        fs.download_batch_from_hf(batch_idx=batch_idx)
        print(f"✅ Downloaded batch {batch_idx}")

    print("✅ Downloaded multiple batches (~750GB+)")


def download_different_splits():
    """
    Download data from different splits and labels.

    Covers both improvised/naturalistic and train/dev/test splits.
    """
    # Download from different combinations
    splits_to_download = [
        ("improvised", "dev", 0),
        ("naturalistic", "dev", 0),
        ("improvised", "test", 0),
        ("naturalistic", "test", 0),
    ]

    for label, split, batch_idx in splits_to_download:
        config = DatasetConfig(label=label, num_workers=4)
        fs = SeamlessInteractionFS(config=config)

        # Download only first few archives to keep size manageable (~1GB per split)
        fs.download_batch_from_hf(
            split=split, batch_idx=batch_idx, archive_list=[0, 1, 2]
        )
        print(f"✅ Downloaded {label}/{split} sample")

    print("✅ Downloaded samples from different splits")


def download_whole_dataset():
    """
    Download the complete dataset (~27TB).

    ⚠️ CAUTION: This will download the entire dataset!
    Only use on high-capacity storage with fast internet.
    """
    # Method 1: Using batch-by-batch download (recommended for control)
    labels = ["improvised", "naturalistic"]
    splits = ["train", "dev", "test"]

    confirm = input(
        "Are you sure you want to download the entire dataset (~27TB)? (y/n): "
    )
    if confirm not in ["y", "Y", "yes", "Yes", "YES"]:
        print("Download cancelled.")
        return

    for label in labels:
        for split in splits:
            print(f"Downloading all {label}/{split} batches...")
            config = DatasetConfig(label=label, num_workers=16)
            fs = SeamlessInteractionFS(config=config)
            fs.download_batch_from_hf(
                split=split,
                batch_idx=None,  # Download all batches
            )

    # Method 2: Using HuggingFace snapshot (alternative)
    # from huggingface_hub import snapshot_download
    # snapshot_download(
    #     repo_id="facebook/seamless-interaction",
    #     repo_type="dataset",
    #     local_dir="~/datasets/seamless_interaction_full"
    # )

    print("✅ Downloaded complete dataset (~27TB)")




def download_balanced():
    """
    Minimal hardcoded flow:
    - has_imitator_movement == 1
    - train only, limited to batches 1..15
    - keep only complete interactions (both participants)
    - balance 50/50 improvised vs naturalistic
    - download required HF archives via fs.download_batch_from_hf
    """
    import pandas as pd

    seed = 42
    df = pd.read_csv("assets/filelist.csv")

    # Scope + validity filter
    df = df[
        (df["split"] == "train")
        & (df["batch_idx"] >= 0)
        & (df["batch_idx"] <= 15)  # Pass 1: batches 0-15. After download, prune (delete MP4s + resample), then change back to 29 for pass 2.
        & (df["has_imitator_movement"] == 1)
    ].copy()

    # Interaction id + keep only complete interactions (>=2 participants)
    df["interaction_id"] = df["file_id"].str.replace(r"_P[0-9A-Za-z]+$", "", regex=True)
    complete = (
        df.groupby(["label", "split", "interaction_id"])["file_id"]
        .nunique()
        .reset_index(name="n")
    )
    complete = complete[complete["n"] >= 2][["label", "split", "interaction_id"]]
    df = df.merge(complete, on=["label", "split", "interaction_id"], how="inner")

    # 50/50 balance per split at interaction level
    picks = []
    for split in ["train"]:
        imp = (
            df[(df["split"] == split) & (df["label"] == "improvised")]["interaction_id"]
            .drop_duplicates()
            .sample(frac=1, random_state=seed)
            .tolist()
        )
        nat = (
            df[(df["split"] == split) & (df["label"] == "naturalistic")]["interaction_id"]
            .drop_duplicates()
            .sample(frac=1, random_state=seed)
            .tolist()
        )
        n = min(len(imp), len(nat))
        picks.append(pd.DataFrame({"split": split, "label": "improvised", "interaction_id": imp[:n]}))
        picks.append(pd.DataFrame({"split": split, "label": "naturalistic", "interaction_id": nat[:n]}))

    picked = pd.concat(picks, ignore_index=True)
    final_df = df.merge(picked, on=["split", "label", "interaction_id"], how="inner")

    # Download only needed archives
    plan = (
        final_df[["label", "split", "batch_idx", "archive_idx"]]
        .drop_duplicates()
        .sort_values(["label", "split", "batch_idx", "archive_idx"])
    )

    for label in ["improvised", "naturalistic"]:
        fs = SeamlessInteractionFS(
            config=DatasetConfig(label=label, preferred_vendors_only=False, num_workers=8)
        )
        sub = plan[plan["label"] == label]
        for split in ["train"]:
            sub_split = sub[sub["split"] == split]
            for batch_idx, group in sub_split.groupby("batch_idx"):
                fs.download_batch_from_hf(
                    split=split,
                    batch_idx=int(batch_idx),
                    archive_list=sorted(group["archive_idx"].astype(int).tolist()),
                )




def main():
    """
    Demonstrate HuggingFace-based flexible download options.
    """
    print("📦 HuggingFace Download Options:")
    print("1. Sample set (~1GB) - Traditional archive-based")
    print("2. Single batch (~50-100GB)")
    print("3. Multiple batches (~150GB+)")
    print("4. Different splits (improvised/naturalistic, train/dev/test)")
    print("5. Whole dataset (~27TB)")

    # Uncomment desired download scenario:
    #download_1gb_sample_archive()
    # download_single_batch()
    # download_multiple_batches()
    # download_different_splits()
    # download_whole_dataset()  # ⚠️ CAUTION: Very large!
    download_balanced()  # Balanced subset with specific criteria


if __name__ == "__main__":
    main()
