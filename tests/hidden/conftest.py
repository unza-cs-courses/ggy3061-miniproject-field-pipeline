"""
Pytest configuration for Mini-Project hidden tests.
"""

import json
import sys
import importlib.util
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')


# Path setup
SRC_DIR = Path(__file__).parent.parent.parent / "src"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
ROOT_DIR = Path(__file__).parent.parent.parent

sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def field_data_path():
    """Return path to the field_samples.csv data file."""
    return DATA_DIR / "field_samples.csv"


@pytest.fixture(scope="session")
def field_dataframe(field_data_path):
    """Load the actual field samples CSV as a DataFrame."""
    return pd.read_csv(str(field_data_path))


@pytest.fixture(scope="session")
def field_csv_file(field_data_path):
    """Return the string path to the field samples CSV file."""
    return str(field_data_path)


@pytest.fixture(scope="session")
def variant_config():
    """Load student's variant configuration from .variant_config.json or generate it."""
    config_path = ROOT_DIR / ".variant_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    # Fall back to generating via get_variant.py
    script_path = ROOT_DIR / "scripts" / "get_variant.py"
    if script_path.exists():
        spec = importlib.util.spec_from_file_location("get_variant", str(script_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.get_my_variant()

    # Default fallback
    return {
        "parameters": {
            "study_area": "Northern Zone",
            "target_elements": ["Au", "Cu", "Pb"],
            "collector_filter": "A. Smith",
            "elevation_range": {"min": 1200, "max": 1600},
            "anomaly_percentile": 95,
            "num_samples": 800
        }
    }


@pytest.fixture
def alternative_dataframe():
    """Create an alternative DataFrame to catch hardcoded values."""
    np.random.seed(99)
    n = 25
    data = {
        'sample_id': [f'ALT-{i:04d}' for i in range(1, n + 1)],
        'collection_date': pd.date_range('2025-06-01', periods=n, freq='3D'),
        'collector': np.random.choice(['A. Smith', 'B. Johnson', 'C. Williams', 'D. Brown'], n),
        'utm_east': np.random.uniform(540000, 560000, n),
        'utm_north': np.random.uniform(7190000, 7210000, n),
        'elevation': np.random.uniform(1100, 1800, n),
        'rock_type': np.random.choice(['Granite', 'Diorite', 'Schist', 'Gneiss', 'Quartzite'], n),
        'alteration': np.random.choice(['Fresh', 'Weak', 'Moderate', 'Strong', 'Intense'], n),
        'structure': np.random.choice(['Massive', 'Foliated', 'Veined', 'Brecciated'], n),
        'Au_ppb': np.random.lognormal(4.5, 1.2, n).round(1),
        'Cu_ppm': np.random.lognormal(3.8, 0.9, n).round(1),
        'Pb_ppm': np.random.lognormal(4.0, 0.6, n).round(1),
        'Zn_ppm': np.random.lognormal(4.5, 0.8, n).round(1),
        'As_ppm': np.random.lognormal(3.0, 1.0, n).round(1),
        'sample_weight': np.random.uniform(300, 1500, n).round(1),
        'qc_flag': np.random.choice(['Passed', 'Failed'], n, p=[0.85, 0.15]),
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary directory for output files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Autouse fixture to clean up matplotlib figures after each test."""
    import matplotlib.pyplot as plt
    yield
    plt.close('all')
