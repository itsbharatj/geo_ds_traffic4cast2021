# run_tests.py
"""
Test runner script
"""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run all tests"""
    
    test_dir = Path(__file__).parent / "tests"
    
    # Run pytest
    cmd = [
        sys.executable, "-m", "pytest", 
        str(test_dir),
        "-v",
        "--tb=short"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed with return code {e.returncode}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)