#!/usr/bin/env python3
"""
Download ETT (Electricity Transformer Temperature) dataset.

Source: https://github.com/zhouhaoyi/ETDataset
Download options:
  1. Direct CSV download from GitHub
  2. Via huggingface datasets library (alternative)

Usage:
    python scripts/download_data.py                     # Download all variants
    python scripts/download_data.py --variant h1        # Download only ETTh1
    python scripts/download_data.py --variant m1        # Download only ETTm1
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import io
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "ETT"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed" / "ETT"

# ETT dataset URLs (from official GitHub)
ETT_URLS = {
    "h1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "h2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "m1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "m2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}

# Alternative: HuggingFace datasets (commented out for simplicity)
# from datasets import load_dataset

ETT_DESCRIPTIONS = {
    "h1": "ETTh1 - Hourly data from transformer 1 (Oil Temperature target)",
    "h2": "ETTh2 - Hourly data from transformer 2 (Oil Temperature target)",
    "m1": "ETTm1 - 15-minute data from transformer 1 (Oil Temperature target)",
    "m2": "ETTm2 - 15-minute data from transformer 2 (Oil Temperature target)",
}


def download_file(url: str, dest_path: Path) -> bool:
    """
    Download a file from URL with progress reporting.
    
    Args:
        url: Source URL
        dest_path: Destination file path
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading: {url}")
        logger.info(f"  -> {dest_path}")
        
        # Create parent directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def report_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                sys.stdout.write(f"\r  Progress: {percent:.1f}% ({downloaded/1024:.1f} KB / {total_size/1024:.1f} KB)")
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\r  Downloaded: {downloaded/1024:.1f} KB")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, report_hook)
        print()  # Newline after progress
        logger.info(f"  ✓ Downloaded successfully ({dest_path.stat().st_size/1024:.1f} KB)")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Failed to download: {e}")
        return False


def download_ett_variant(variant: str, force: bool = False) -> bool:
    """
    Download a specific ETT variant.
    
    Args:
        variant: 'h1', 'h2', 'm1', or 'm2'
        force: If True, redownload even if file exists
    
    Returns:
        bool: True if successful
    """
    if variant not in ETT_URLS:
        logger.error(f"Unknown variant: {variant}. Available: {list(ETT_URLS.keys())}")
        return False
    
    dest_path = RAW_DATA_DIR / f"ETT{variant[0]}{variant[1]}.csv"
    
    if dest_path.exists() and not force:
        logger.info(f"  {variant}: File already exists at {dest_path} (use --force to redownload)")
        return True
    
    logger.info(f"\n=== Downloading {variant} ===")
    logger.info(f"Description: {ETT_DESCRIPTIONS[variant]}")
    
    return download_file(ETT_URLS[variant], dest_path)


def validate_dataset(variant: str) -> bool:
    """
    Validate that the downloaded CSV is readable and has expected columns.
    
    Args:
        variant: 'h1', 'h2', 'm1', or 'm2'
    
    Returns:
        bool: True if valid
    """
    try:
        import pandas as pd
        
        file_path = RAW_DATA_DIR / f"ETT{variant[0]}{variant[1]}.csv"
        
        if not file_path.exists():
            logger.error(f"  {variant}: File not found")
            return False
        
        # Try reading the CSV
        df = pd.read_csv(file_path)
        
        # Expected columns for ETT
        expected_cols = ['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        
        if not all(col in df.columns for col in expected_cols):
            logger.warning(f"  {variant}: Unexpected columns. Found: {df.columns.tolist()}")
        
        logger.info(f"  {variant}: ✓ Valid - {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"  Columns: {', '.join(df.columns[:4])}... (total {len(df.columns)})")
        logger.info(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
        
        return True
        
    except ImportError:
        logger.warning("  pandas not installed, skipping validation")
        return True
    except Exception as e:
        logger.error(f"  {variant}: Validation failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download ETT dataset for time series forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all variants
  python scripts/download_data.py
  
  # Download specific variant
  python scripts/download_data.py --variant h1
  
  # Force redownload
  python scripts/download_data.py --force
        """
    )
    
    parser.add_argument(
        "--variant", "-v",
        choices=list(ETT_URLS.keys()),
        help="Download specific variant (h1, h2, m1, m2). If not specified, download all."
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force redownload even if files exist"
    )
    
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Validate downloaded files"
    )
    
    args = parser.parse_args()
    
    # Create directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine which variants to download
    variants = [args.variant] if args.variant else list(ETT_URLS.keys())
    
    logger.info("=" * 60)
    logger.info("ETT Dataset Downloader")
    logger.info("=" * 60)
    logger.info(f"Raw data directory: {RAW_DATA_DIR}")
    logger.info(f"Processed data directory: {PROCESSED_DATA_DIR}")
    logger.info(f"Variants to download: {', '.join(variants)}")
    logger.info("=" * 60)
    
    # Download each variant
    success_count = 0
    for variant in variants:
        if download_ett_variant(variant, args.force):
            success_count += 1
            if args.validate:
                validate_dataset(variant)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"Download summary: {success_count}/{len(variants)} variants downloaded successfully")
    logger.info(f"Files location: {RAW_DATA_DIR}")
    logger.info("=" * 60)
    
    if success_count == len(variants):
        logger.info("✓ All downloads complete!")
        
        # Show next steps
        logger.info("\nNext steps:")
        logger.info("  1. Validate data: python scripts/validate_data.py")
        logger.info("  2. Start EDA: jupyter notebook notebooks/01_eda_and_data_preprocessing.ipynb")
        logger.info("  3. Train models: python src/train.py")
    else:
        logger.warning("Some downloads failed. You may need to:")
        logger.warning("  - Check your internet connection")
        logger.warning("  - Try again with --force")
        logger.warning("  - Alternative: Use huggingface datasets library")
        sys.exit(1)


if __name__ == "__main__":
    main()
