"""
Quick test script to verify all modules can be imported and basic functionality works
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from initialisation import *
        print("✓ initialisation.py imported successfully")
        print(f"  - Period: {period}")
        print(f"  - Alpha: {alpha}")
        print(f"  - Stocks to process: {len(stocks)}")
    except Exception as e:
        print(f"✗ Failed to import initialisation: {e}")
        return False
    
    try:
        from api_yahoo_finance import download_stock
        print("✓ api_yahoo_finance.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import api_yahoo_finance: {e}")
        return False
    
    try:
        from utility import log_returns, discrete_returns
        print("✓ utility.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import utility: {e}")
        return False
    
    try:
        from traditional_models import apply_traditional_models
        print("✓ traditional_models.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import traditional_models: {e}")
        return False
    
    try:
        from GARCH_models import apply_garch_models
        print("✓ GARCH_models.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import GARCH_models: {e}")
        return False
    
    try:
        from backtesting import run_backtesting
        print("✓ backtesting.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import backtesting: {e}")
        return False
    
    try:
        from NNet_sourcer import load_nnet_results
        print("✓ NNet_sourcer.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import NNet_sourcer: {e}")
        return False
    
    try:
        from Gaussian_mixture_sampling import r_gaussmix_2c, r_gaussmix_3c
        print("✓ Gaussian_mixture_sampling.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import Gaussian_mixture_sampling: {e}")
        return False
    
    return True

def test_dependencies():
    """Test that all required packages are installed"""
    print("\nTesting dependencies...")
    
    packages = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('scipy', None),
        ('yfinance', 'yf'),
        ('arch', None),
    ]
    
    all_ok = True
    for package, alias in packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} NOT installed")
            all_ok = False
    
    # Test TensorFlow and TensorFlow Probability (optional)
    try:
        import tensorflow as tf
        print(f"✓ tensorflow installed (version {tf.__version__})")
    except ImportError:
        print("⚠ tensorflow NOT installed (optional, needed for LSTM-MDN training)")
    
    try:
        import tensorflow_probability as tfp
        print(f"✓ tensorflow_probability installed")
    except ImportError:
        print("⚠ tensorflow_probability NOT installed (optional, needed for LSTM-MDN training)")
    
    try:
        import tf_keras
        print(f"✓ tf_keras installed")
    except ImportError:
        print("⚠ tf_keras NOT installed (optional, needed for LSTM-MDN training)")
    
    return all_ok

def test_data_access():
    """Test that data files can be accessed"""
    print("\nTesting data access...")
    
    from pathlib import Path
    import pandas as pd
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    
    csv_path = DATA_DIR / "nifty100_constituents.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"✓ nifty100_constituents.csv found ({len(df)} stocks)")
        print(f"  First stock: {df['Symbol'].iloc[0]}")
    else:
        print(f"✗ nifty100_constituents.csv NOT found at {csv_path}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    import numpy as np
    from utility import log_returns, discrete_returns
    
    # Test return calculations
    prices = np.array([100, 102, 101, 103, 105])
    
    log_ret = log_returns(prices)
    disc_ret = discrete_returns(prices)
    
    print(f"✓ log_returns works (first non-NA: {log_ret[1]:.6f})")
    print(f"✓ discrete_returns works (first non-NA: {disc_ret[1]:.6f})")
    
    # Test Gaussian mixture sampling
    from Gaussian_mixture_sampling import r_gaussmix_2c
    
    samples = r_gaussmix_2c(
        N=1000,
        mu1=0.0, mu2=0.1,
        sigma1=0.01, sigma2=0.02,
        pi1=0.5, pi2=0.5
    )
    
    print(f"✓ Gaussian mixture sampling works (mean: {np.mean(samples):.6f})")
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("NIFTY100 VaR Project - Test Suite")
    print("="*60)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_dependencies()
    all_tests_passed &= test_data_access()
    all_tests_passed &= test_imports()
    all_tests_passed &= test_basic_functionality()
    
    print("\n" + "="*60)
    if all_tests_passed:
        print("✓ ALL TESTS PASSED")
        print("\nYou can now run: python execution.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease install missing dependencies:")
        print("  pip install -r ../requirements.txt")
    print("="*60)
