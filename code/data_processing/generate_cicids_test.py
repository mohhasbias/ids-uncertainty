#!/usr/bin/env python3
"""
Generate CIC-IDS2017 test dataset in JSONL format

Converts stratified sample of CIC-IDS2017 CSV to natural language descriptions
compatible with existing LLM-based IDS experiments.

Output format matches Mock IDS Hard dataset structure.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from cicids_converter import CICIDSConverter, stratified_sample_cicids


def convert_flow_to_jsonl_entry(
    flow: pd.Series,
    sample_id: int,
    converter: CICIDSConverter,
    original_index: int,
    source_label: str = "CIC-IDS2017",
    dataset_label: str = "cicids2017_cleaned",
    attack_column: str = "Attack Type"
) -> Dict[str, Any]:
    """
    Convert a single flow (pandas Series) to JSONL entry format

    Args:
        flow: Pandas Series with network flow features
        sample_id: Sequential ID for this sample
        converter: CICIDSConverter instance
        original_index: Original index in CIC-IDS dataset
        source_label: Source dataset label for metadata (default: "CIC-IDS2017")
        dataset_label: Dataset identifier for metadata (default: "cicids2017_cleaned")
        attack_column: Name of attack/label column (default: "Attack Type")

    Returns:
        Dictionary in JSONL format
    """
    # Convert Series to dict for converter
    flow_dict = flow.to_dict()

    # Generate natural language description
    description = converter.convert_flow_to_text(flow_dict)

    # Get original attack type/label and normalize it
    original_attack = flow[attack_column]

    # Normalize attack type using converter (handles both numeric and text labels)
    # For CIC-IDS2018: numeric (1-11) → attack names → standard categories
    # For CIC-IDS2017: text labels → standard categories
    attack_type = converter.normalize_attack_type(original_attack)

    # Determine label (malicious or benign)
    label = attack_type if attack_type != 'benign' else 'benign'

    # Create features dict (original numerical values, exclude the attack/label column)
    features = {k: v for k, v in flow_dict.items() if k != attack_column}

    # Build JSONL entry
    entry = {
        "id": f"cicids_{sample_id}",
        "description": description,
        "label": label,
        "attack_type": attack_type,
        "difficulty": "medium",  # Real-world data = medium difficulty
        "features": features,
        "metadata": {
            "source": source_label,
            "original_index": int(original_index),
            "dataset": dataset_label,
            "original_attack_type": original_attack
        }
    }

    return entry


def generate_cicids_test_dataset(
    input_csv: str = "datasets/cicids2017/cicids2017_cleaned.csv",
    output_jsonl: str = "datasets/cicids2017_test_50.jsonl",
    n_samples: int = 50,
    random_state: int = 42,
    source_label: str = "CIC-IDS2017",
    dataset_label: str = "cicids2017_cleaned",
    dataset_version: str = "2017",
    attack_column: str = "Attack Type"
) -> None:
    """
    Generate test dataset from CIC-IDS

    Args:
        input_csv: Path to CIC-IDS CSV file
        output_jsonl: Path to output JSONL file
        n_samples: Number of samples to generate (default 50)
        random_state: Random seed for reproducibility
        source_label: Source dataset label for metadata (default: "CIC-IDS2017")
        dataset_label: Dataset identifier for metadata (default: "cicids2017_cleaned")
        dataset_version: Dataset version '2017' or '2018' (default: "2017")
        attack_column: Name of attack/label column (default: "Attack Type")
    """
    print("=" * 70)
    print(f"{source_label} Test Dataset Generation")
    print("=" * 70)

    # Load CIC-IDS dataset
    print(f"\n[1/5] Loading {source_label} dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"   ✓ Loaded {len(df):,} flows with {len(df.columns)} features")

    # Display original attack distribution
    print(f"\n[2/5] Original attack distribution:")
    for attack_type, count in df[attack_column].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"   {attack_type}: {count:,} ({percentage:.1f}%)")

    # Perform stratified sampling
    print(f"\n[3/5] Performing stratified sampling ({n_samples} samples)...")
    sampled_df = stratified_sample_cicids(
        df,
        n_samples=n_samples,
        attack_column=attack_column,
        random_state=random_state,
        dataset_version=dataset_version
    )

    # Convert to natural language descriptions
    print(f"\n[4/5] Converting flows to natural language descriptions...")
    converter = CICIDSConverter()
    jsonl_entries = []

    for idx, (original_idx, flow) in enumerate(sampled_df.iterrows(), start=1):
        entry = convert_flow_to_jsonl_entry(
            flow=flow,
            sample_id=idx,
            converter=converter,
            original_index=original_idx,
            source_label=source_label,
            dataset_label=dataset_label,
            attack_column=attack_column
        )
        jsonl_entries.append(entry)

        # Print progress every 10 samples
        if idx % 10 == 0:
            print(f"   Converted {idx}/{n_samples} samples...")

    print(f"   ✓ Converted all {n_samples} samples")

    # Save to JSONL file
    print(f"\n[5/5] Saving to JSONL file: {output_jsonl}")
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"   ✓ Saved {len(jsonl_entries)} entries to {output_jsonl}")

    # Generate statistics report
    print("\n" + "=" * 70)
    print("Generation Complete - Summary Statistics")
    print("=" * 70)

    # Attack type distribution
    attack_counts = {}
    for entry in jsonl_entries:
        attack_type = entry['attack_type']
        attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1

    print(f"\nFinal dataset distribution ({len(jsonl_entries)} samples):")
    for attack_type, count in sorted(attack_counts.items(), key=lambda x: -x[1]):
        percentage = (count / len(jsonl_entries)) * 100
        print(f"   {attack_type}: {count} samples ({percentage:.1f}%)")

    # Calculate description statistics
    desc_lengths = [len(entry['description']) for entry in jsonl_entries]
    avg_length = sum(desc_lengths) / len(desc_lengths)
    min_length = min(desc_lengths)
    max_length = max(desc_lengths)

    print(f"\nDescription statistics:")
    print(f"   Average length: {avg_length:.0f} characters")
    print(f"   Min length: {min_length} characters")
    print(f"   Max length: {max_length} characters")

    # Estimate token usage (rough estimate: 1 token ≈ 4 characters)
    avg_tokens = avg_length / 4
    print(f"   Estimated tokens per description: ~{avg_tokens:.0f} tokens")

    # Show 3 example descriptions
    print("\n" + "=" * 70)
    print("Example Descriptions (First 3 Samples)")
    print("=" * 70)

    for i, entry in enumerate(jsonl_entries[:3], start=1):
        print(f"\n[Sample {i}] ID: {entry['id']}")
        print(f"Attack Type: {entry['attack_type']}")
        print(f"Description: {entry['description']}")
        print(f"Character count: {len(entry['description'])}")

    print("\n" + "=" * 70)
    print("Dataset generation complete!")
    print("=" * 70)
    print(f"\nOutput file: {output_path.absolute()}")
    print(f"Ready for use with baseline_runner.py and ensemble_runner.py")
    print("\nNext steps:")
    print("   1. Validate with: python dataset_loader.py")
    print("   2. Run 5-sample trial: python baseline_runner.py --dataset cicids2017_test_50 --samples 5")
    print("   3. Check API costs before full run")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate CIC-IDS test dataset in JSONL format"
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        choices=["2017", "2018"],
        default="2017",
        help="CIC-IDS dataset version (default: 2017)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to CIC-IDS CSV file (overrides default for dataset version)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSONL file (overrides default for dataset version)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to generate (default: 50)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Configure paths and metadata based on dataset version
    if args.dataset_version == "2018":
        default_input = "datasets/cicids2018/cleaned_ids2018_sampled.csv"
        default_output = "datasets/cicids2018_test_50.jsonl"
        source_label = "CIC-IDS2018"
        dataset_label = "cicids2018_cleaned"
        attack_column = "Label"
    else:  # 2017
        default_input = "datasets/cicids2017/cicids2017_cleaned.csv"
        default_output = "datasets/cicids2017_test_50.jsonl"
        source_label = "CIC-IDS2017"
        dataset_label = "cicids2017_cleaned"
        attack_column = "Attack Type"

    # Allow command-line overrides of defaults
    input_csv = args.input if args.input is not None else default_input
    output_jsonl = args.output if args.output is not None else default_output

    generate_cicids_test_dataset(
        input_csv=input_csv,
        output_jsonl=output_jsonl,
        n_samples=args.samples,
        random_state=args.seed,
        source_label=source_label,
        dataset_label=dataset_label,
        dataset_version=args.dataset_version,
        attack_column=attack_column
    )
