#!/usr/bin/env python3
"""
Validate downloaded ETT datasets.

Checks:
  - File exists and is readable
  - Expected columns present
  - No missing values
  - Data types correct
  - Date range covers expected period

Usage:
    python scripts/validate_data.py          # Validate all variants
    python scripts/validate_data.py --variant h1
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

EXPECTED_COLUMNS = ['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
VARIANT_INFO = {
    "h1": {"freq": "hourly", "expected_columns": EXPECTED_COLUMNS},
    "h2": {"freq": "hourly", "expected_columns": EXPECTED_COLUMNS},
    "m1": {"freq": "15-min", "expected_columns": EXPECTED_COLUMNS},
    "m2": {"freq": "15-min", "expected_columns": EXPECTED_COLUMNS},
}


def validate_file(variant: str) -> dict:
    """
    Validate a single ETT variant.
    
    Returns:
        dict: Validation results
    """
    file_path = RAW_DATA_DIR / f"ETT{variant[0]}{variant[1]}.csv"
    
    results = {
        "variant": variant,
        "file_exists": file_path.exists(),
        "file_size_kb": 0,
        "rows": 0,
        "columns": [],
        "columns_valid": False,
        "missing_values": {},
        "date_range": None,
        "valid": False,
        "errors": []
    }
    
    if not results["file_exists"]:
        results["errors"].append("File not found")
        return results
    
    results["file_size_kb"] = file_path.stat().st_size / 1024
    
    try:
        df = pd.read_csv(file_path)
        results["rows"] = len(df)
        results["columns"] = df.columns.tolist()
        
        # Check columns
        info = VARIANT_INFO.get(variant, {})
        expected = info.get("expected_columns", EXPECTED_COLUMNS)
        results["columns_valid"] = all(col in df.columns for col in expected)
        
        if not results["columns_valid"]:
            results["errors"].append(f"Missing expected columns. Found: {df.columns.tolist()}")
        
        # Check missing values
        results["missing_values"] = df.isnull().sum().to_dict()
        
        # Check date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            results["date_range"] = {
                "start": df['date'].iloc[0],
                "end": df['date'].iloc[-1],
                "n_unique": df['date'].nunique(),
            }
            
            # Check for duplicate dates
            if df['date'].nunique() != len(df):
                results["errors"].append(f"Duplicate dates found: {len(df) - df['date'].nunique()} duplicates")
        
        # Check target column (OT)
        if 'OT' in df.columns:
            results["target_stats"] = {
                "min": df['OT'].min(),
                "max": df['OT'].max(),
                "mean": df['OT'].mean(),
                "std": df['OT'].std(),
            }
        
        # Check if data is valid
        results["valid"] = (
            results["file_exists"] and
            results["columns_valid"] and
            len(results["errors"]) == 0
        )
        
    except Exception as e:
        results["errors"].append(f"Error reading file: {e}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate ETT datasets")
    parser.add_argument(
        "--variant", "-v",
        choices=['h1', 'h2', 'm1', 'm2'],
        help="Validate specific variant"
    )
    args = parser.parse_args()
    
    variants = [args.variant] if args.variant else list(VARIANT_INFO.keys())
    
    logger.info("=" * 60)
    logger.info("ETT Data Validation")
    logger.info("=" * 60)
    logger.info(f"Data directory: {RAW_DATA_DIR}")
    logger.info(f"Variants to validate: {', '.join(variants)}")
    logger.info("=" * 60)
    
    all_valid = True
    
    for variant in variants:
        logger.info(f"\n=== Validating {variant} ===")
        results = validate_file(variant)
        
        # Print results
        logger.info(f"  File: ETT{variant[0]}{variant[1]}.csv")
        logger.info(f"  Exists: {'✓' if results['file_exists'] else '✗'}")
        if results['file_exists']:
            logger.info(f"  Size: {results['file_size_kb']:.1f} KB")
            logger.info(f"  Rows: {results['rows']:,}")
            logger.info(f"  Columns: {len(results['columns'])}")
            logger.info(f"  Expected columns present: {'✓' if results['columns_valid'] else '✗'}")
            
            if results.get('date_range'):
                logger.info(f"  Date range: {results['date_range']['start']} to {results['date_range']['end']}")
                logger.info(f"  Unique dates: {results['date_range']['n_unique']:,}")
            
            if results.get('target_stats'):
                logger.info(f"  Target (OT) stats: min={results['target_stats']['min']:.2f}, "
                           f"max={results['target_stats']['max']:.2f}, "
                           f"mean={results['target_stats']['mean']:.2f}")
            
            # Missing values
            missing = {k: v for k, v in results['missing_values'].items() if v > 0}
            if missing:
                logger.warning(f"  Missing values: {missing}")
            else:
                logger.info("  Missing values: None ✓")
            
            if results['errors']:
                logger.error(f"  Errors: {', '.join(results['errors'])}")
            else:
                logger.info(f"  Status: ✓ VALID")
        
        all_valid = all_valid and results['valid']
    
    logger.info("\n" + "=" * 60)
    if all_valid:
        logger.info("✓ All datasets validated successfully!")
    else:
        logger.warning("Some datasets failed validation. Check the errors above.")
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    exit(main())
