import os
import sys


def check_directory_structure():
    """Verify that all expected directories exist."""
    required_dirs = [
        'data/raw', 'data/processed', 'data/forecasts',
        'notebooks', 'src', 'configs', 'models',
        'tests', 'docs', 'scripts', 'logs'
    ]

    print("Checking directory structure...")
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  [OK]     {dir_path}/")
        else:
            print(f"  [MISSING] {dir_path}/")
            all_exist = False
    return all_exist


if __name__ == "__main__":
    print("Project Directory Check")
    print("=" * 50)

    if check_directory_structure():
        print("\nAll directories present.")
        print(f"Project location: {os.getcwd()}")
        print("\nNext steps:")
        print("  1. Install requirements: make install")
        print("  2. Download data:        python scripts/download_data.py")
        print("  3. Start a notebook:     jupyter notebook notebooks/")
    else:
        print("\nSome directories are missing. Run the setup script again.")
        sys.exit(1)
