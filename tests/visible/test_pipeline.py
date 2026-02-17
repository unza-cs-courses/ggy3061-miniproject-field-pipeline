"""
Mini-Project Visible Tests - Field Sample Pipeline
"""

import subprocess
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

SRC_DIR = Path(__file__).parent.parent.parent / "src"
DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Conditionally import modules - they may not be implemented yet
sys.path.insert(0, str(SRC_DIR))

# Try importing loader functions
try:
    from pipeline.loader import load_samples, validate_data, get_loading_statistics, load_and_validate
    HAS_LOADER = True
except ImportError:
    HAS_LOADER = False

# Try importing cleaner functions
try:
    from pipeline.cleaner import (
        handle_missing_values, detect_outliers, filter_by_criteria,
        clean_data, standardize_data_types, handle_outliers as cleaner_handle_outliers
    )
    HAS_CLEANER = True
except ImportError:
    HAS_CLEANER = False

# Try importing analyzer functions
try:
    from pipeline.analyzer import (
        calculate_statistics, group_statistics, correlation_analysis,
        identify_significant_correlations, analyze_by_spatial_region,
        generate_analysis_summary
    )
    HAS_ANALYZER = True
except ImportError:
    HAS_ANALYZER = False

# Try importing detector functions
try:
    from pipeline.detector import (
        detect_anomalies, calculate_thresholds, AnomalyDetector,
        get_anomaly_spatial_context, summarize_anomalies
    )
    HAS_DETECTOR = True
except ImportError:
    HAS_DETECTOR = False

# Try importing reporter functions
try:
    from pipeline.reporter import (
        generate_text_report, generate_html_report, save_report,
        create_executive_summary
    )
    HAS_REPORTER = True
except ImportError:
    HAS_REPORTER = False

# Try importing visualizer functions
try:
    from pipeline.visualizer import (
        apply_professional_style, plot_element_histograms,
        plot_spatial_distribution, plot_anomaly_map,
        plot_correlation_heatmap, create_summary_plot, save_figure
    )
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False

# Try importing main pipeline
try:
    sys.path.insert(0, str(SRC_DIR))
    from main import run_pipeline
    HAS_MAIN = True
except ImportError:
    HAS_MAIN = False


# ============================================================================
# STRUCTURAL TESTS (keep existing tests)
# ============================================================================

class TestPipelineStructure:
    """Tests for pipeline module structure."""

    def test_pipeline_package_exists(self):
        """pipeline package should exist."""
        assert (SRC_DIR / "pipeline").exists()

    def test_loader_module_exists(self):
        """loader.py should exist."""
        assert (SRC_DIR / "pipeline" / "loader.py").exists()

    def test_cleaner_module_exists(self):
        """cleaner.py should exist."""
        assert (SRC_DIR / "pipeline" / "cleaner.py").exists()

    def test_analyzer_module_exists(self):
        """analyzer.py should exist."""
        assert (SRC_DIR / "pipeline" / "analyzer.py").exists()

    def test_detector_module_exists(self):
        """detector.py should exist."""
        assert (SRC_DIR / "pipeline" / "detector.py").exists()

    def test_visualizer_module_exists(self):
        """visualizer.py should exist."""
        assert (SRC_DIR / "pipeline" / "visualizer.py").exists()

    def test_reporter_module_exists(self):
        """reporter.py should exist."""
        assert (SRC_DIR / "pipeline" / "reporter.py").exists()


class TestMainScript:
    """Tests for main pipeline script."""

    def test_main_exists(self):
        """main.py should exist."""
        assert (SRC_DIR / "main.py").exists()


class TestDataFiles:
    """Tests for data files."""

    def test_field_samples_exists(self):
        """field_samples.csv should exist."""
        assert (DATA_DIR / "field_samples.csv").exists()


# ============================================================================
# FIXTURES - Create shared test data
# ============================================================================

@pytest.fixture
def sample_csv(tmp_path):
    """Create a small test CSV with field sample data."""
    np.random.seed(42)
    n = 15
    data = {
        'sample_id': [f'S{i:03d}' for i in range(1, n+1)],
        'utm_east': np.random.uniform(500000, 510000, n),
        'utm_north': np.random.uniform(8200000, 8210000, n),
        'elevation': np.random.uniform(1000, 1500, n),
        'Au_ppb': np.random.lognormal(0, 1, n).round(3),
        'Cu_ppm': np.random.lognormal(4, 0.8, n).round(1),
        'Pb_ppm': np.random.lognormal(3, 0.5, n).round(1),
        'Zn_ppm': np.random.lognormal(4, 0.7, n).round(1),
        'As_ppm': np.random.lognormal(2, 1, n).round(1),
        'collector': np.random.choice(['A. Smith', 'B. Johnson', 'C. Williams'], n),
        'collection_date': pd.date_range('2024-01-01', periods=n, freq='D'),
        'rock_type': np.random.choice(['Granite', 'Schist', 'Diorite'], n),
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_samples.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path), df


@pytest.fixture
def sample_dataframe():
    """Create test DataFrame directly."""
    np.random.seed(42)
    n = 15
    data = {
        'sample_id': [f'S{i:03d}' for i in range(1, n+1)],
        'utm_east': np.random.uniform(500000, 510000, n),
        'utm_north': np.random.uniform(8200000, 8210000, n),
        'elevation': np.random.uniform(1000, 1500, n),
        'Au_ppb': np.random.lognormal(0, 1, n).round(3),
        'Cu_ppm': np.random.lognormal(4, 0.8, n).round(1),
        'Pb_ppm': np.random.lognormal(3, 0.5, n).round(1),
        'Zn_ppm': np.random.lognormal(4, 0.7, n).round(1),
        'As_ppm': np.random.lognormal(2, 1, n).round(1),
        'collector': np.random.choice(['A. Smith', 'B. Johnson', 'C. Williams'], n),
        'collection_date': pd.date_range('2024-01-01', periods=n, freq='D'),
        'rock_type': np.random.choice(['Granite', 'Schist', 'Diorite'], n),
    }
    return pd.DataFrame(data)


# ============================================================================
# LOADER FUNCTION TESTS
# ============================================================================

class TestLoaderFunctions:
    """Tests for data loading and validation functions."""

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not importable")
    def test_load_samples_returns_dataframe(self, sample_csv):
        """load_samples should return a pandas DataFrame."""
        csv_path, _ = sample_csv
        result = load_samples(csv_path)
        assert isinstance(result, pd.DataFrame), "load_samples must return a DataFrame"

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not importable")
    def test_load_samples_correct_rows(self, sample_csv):
        """load_samples should load all rows from CSV."""
        csv_path, expected_df = sample_csv
        result = load_samples(csv_path)
        assert len(result) == len(expected_df), "Incorrect number of rows loaded"

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not importable")
    def test_load_samples_correct_columns(self, sample_csv):
        """load_samples should load all columns from CSV."""
        csv_path, expected_df = sample_csv
        result = load_samples(csv_path)
        assert list(result.columns) == list(expected_df.columns), "Columns mismatch"

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not importable")
    def test_load_samples_file_not_found(self):
        """load_samples should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_samples("/nonexistent/path/to/file.csv")

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not importable")
    def test_validate_data_valid_dataframe(self, sample_dataframe):
        """validate_data should return (True, []) for valid DataFrame."""
        is_valid, errors = validate_data(sample_dataframe)
        assert isinstance(is_valid, bool), "First return value must be bool"
        assert isinstance(errors, list), "Second return value must be list"
        assert is_valid is True, "Should validate correct DataFrame"
        assert len(errors) == 0, "Should have no error messages"

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not importable")
    def test_validate_data_missing_required_column(self, sample_dataframe):
        """validate_data should fail when required column is missing."""
        df = sample_dataframe.drop('sample_id', axis=1)
        is_valid, errors = validate_data(df)
        assert is_valid is False, "Should fail with missing column"
        assert len(errors) > 0, "Should have error messages"

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not importable")
    def test_validate_data_missing_element_column(self, sample_dataframe):
        """validate_data should warn about missing element columns."""
        df = sample_dataframe.drop('Au_ppb', axis=1)
        is_valid, errors = validate_data(df)
        assert is_valid is False, "Should fail with missing element column"

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not importable")
    def test_get_loading_statistics_returns_dict(self, sample_dataframe):
        """get_loading_statistics should return a dictionary."""
        result = get_loading_statistics(sample_dataframe)
        assert isinstance(result, dict), "Should return a dictionary"

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not importable")
    def test_get_loading_statistics_contains_expected_keys(self, sample_dataframe):
        """get_loading_statistics should contain key statistics."""
        result = get_loading_statistics(sample_dataframe)
        expected_keys = ['total_samples', 'unique_collectors', 'missing_values']
        for key in expected_keys:
            assert key in result, f"Missing key '{key}' in statistics"

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not importable")
    def test_get_loading_statistics_correct_sample_count(self, sample_dataframe):
        """get_loading_statistics should report correct sample count."""
        result = get_loading_statistics(sample_dataframe)
        assert result['total_samples'] == len(sample_dataframe)

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not importable")
    def test_get_loading_statistics_unique_collectors(self, sample_dataframe):
        """get_loading_statistics should count unique collectors."""
        result = get_loading_statistics(sample_dataframe)
        expected = sample_dataframe['collector'].nunique()
        assert result['unique_collectors'] == expected

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not importable")
    def test_load_and_validate_returns_tuple(self, sample_csv):
        """load_and_validate should return (DataFrame, dict)."""
        csv_path, _ = sample_csv
        result = load_and_validate(csv_path)
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 2, "Should return 2-element tuple"
        assert isinstance(result[0], pd.DataFrame), "First element should be DataFrame"
        assert isinstance(result[1], dict), "Second element should be dict"

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not importable")
    def test_load_and_validate_dataframe_not_empty(self, sample_csv):
        """load_and_validate should return non-empty DataFrame."""
        csv_path, _ = sample_csv
        df, stats = load_and_validate(csv_path)
        assert len(df) > 0, "DataFrame should not be empty"


# ============================================================================
# CLEANER FUNCTION TESTS
# ============================================================================

class TestCleanerFunctions:
    """Tests for data cleaning and preprocessing functions."""

    @pytest.mark.skipif(not HAS_CLEANER, reason="Cleaner module not importable")
    def test_handle_missing_values_returns_tuple(self, sample_dataframe):
        """handle_missing_values should return (DataFrame, dict)."""
        result = handle_missing_values(sample_dataframe, strategy="median")
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 2, "Should return 2-element tuple"
        assert isinstance(result[0], pd.DataFrame), "First element should be DataFrame"
        assert isinstance(result[1], dict), "Second element should be dict"

    @pytest.mark.skipif(not HAS_CLEANER, reason="Cleaner module not importable")
    def test_handle_missing_values_preserves_shape(self, sample_dataframe):
        """handle_missing_values should preserve DataFrame shape for no NaN case."""
        result_df, _ = handle_missing_values(sample_dataframe, strategy="median")
        assert result_df.shape[0] == sample_dataframe.shape[0] or \
               result_df.shape[0] < sample_dataframe.shape[0], \
               "Row count should not increase"

    @pytest.mark.skipif(not HAS_CLEANER, reason="Cleaner module not importable")
    def test_handle_missing_values_invalid_strategy(self, sample_dataframe):
        """handle_missing_values should raise ValueError for invalid strategy."""
        with pytest.raises(ValueError):
            handle_missing_values(sample_dataframe, strategy="invalid_strategy")

    @pytest.mark.skipif(not HAS_CLEANER, reason="Cleaner module not importable")
    def test_detect_outliers_returns_boolean_dataframe(self, sample_dataframe):
        """detect_outliers should return boolean DataFrame."""
        result = detect_outliers(sample_dataframe, method="iqr")
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert result.shape[0] == sample_dataframe.shape[0], "Should have same row count"
        assert result.dtypes.unique()[0] == bool, "Should contain boolean values"

    @pytest.mark.skipif(not HAS_CLEANER, reason="Cleaner module not importable")
    def test_detect_outliers_iqr_method(self, sample_dataframe):
        """detect_outliers should work with IQR method."""
        result = detect_outliers(sample_dataframe, method="iqr", threshold=1.5)
        assert result is not None, "Should return valid result"

    @pytest.mark.skipif(not HAS_CLEANER, reason="Cleaner module not importable")
    def test_detect_outliers_zscore_method(self, sample_dataframe):
        """detect_outliers should work with zscore method."""
        result = detect_outliers(sample_dataframe, method="zscore", threshold=3.0)
        assert result is not None, "Should return valid result"

    @pytest.mark.skipif(not HAS_CLEANER, reason="Cleaner module not importable")
    def test_filter_by_criteria_returns_dataframe(self, sample_dataframe):
        """filter_by_criteria should return a DataFrame."""
        result = filter_by_criteria(sample_dataframe)
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"

    @pytest.mark.skipif(not HAS_CLEANER, reason="Cleaner module not importable")
    def test_filter_by_criteria_no_filters(self, sample_dataframe):
        """filter_by_criteria with no filters should return all rows."""
        result = filter_by_criteria(sample_dataframe)
        assert len(result) == len(sample_dataframe), "Should return all rows when no filter"

    @pytest.mark.skipif(not HAS_CLEANER, reason="Cleaner module not importable")
    def test_filter_by_criteria_by_collector(self, sample_dataframe):
        """filter_by_criteria should filter by collector."""
        collector = sample_dataframe['collector'].iloc[0]
        result = filter_by_criteria(sample_dataframe, collector=collector)
        assert all(result['collector'] == collector), "Should filter by collector"
        assert len(result) <= len(sample_dataframe), "Filtered result should be smaller"

    @pytest.mark.skipif(not HAS_CLEANER, reason="Cleaner module not importable")
    def test_filter_by_criteria_by_elevation_range(self, sample_dataframe):
        """filter_by_criteria should filter by elevation range."""
        result = filter_by_criteria(
            sample_dataframe,
            elevation_range=(1100, 1400)
        )
        assert all(result['elevation'] >= 1100), "Should respect min elevation"
        assert all(result['elevation'] <= 1400), "Should respect max elevation"

    @pytest.mark.skipif(not HAS_CLEANER, reason="Cleaner module not importable")
    def test_clean_data_returns_tuple(self, sample_dataframe):
        """clean_data should return (DataFrame, dict)."""
        result = clean_data(sample_dataframe)
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 2, "Should return 2-element tuple"
        assert isinstance(result[0], pd.DataFrame), "First element should be DataFrame"
        assert isinstance(result[1], dict), "Second element should be dict"

    @pytest.mark.skipif(not HAS_CLEANER, reason="Cleaner module not importable")
    def test_clean_data_report_contains_keys(self, sample_dataframe):
        """clean_data report should contain key statistics."""
        _, report = clean_data(sample_dataframe)
        expected_keys = ['original_rows', 'final_rows']
        for key in expected_keys:
            assert key in report, f"Missing key '{key}' in cleaning report"

    @pytest.mark.skipif(not HAS_CLEANER, reason="Cleaner module not importable")
    def test_handle_outliers_returns_dataframe(self, sample_dataframe):
        """handle_outliers should return a DataFrame."""
        outlier_mask = pd.DataFrame(
            False, index=sample_dataframe.index,
            columns=['Au_ppb', 'Cu_ppm']
        )
        result = cleaner_handle_outliers(sample_dataframe, outlier_mask, strategy="clip")
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"

    @pytest.mark.skipif(not HAS_CLEANER, reason="Cleaner module not importable")
    def test_handle_outliers_keep_strategy(self, sample_dataframe):
        """handle_outliers with 'keep' strategy should preserve data."""
        outlier_mask = pd.DataFrame(
            False, index=sample_dataframe.index,
            columns=['Au_ppb', 'Cu_ppm']
        )
        result = cleaner_handle_outliers(sample_dataframe, outlier_mask, strategy="keep")
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert len(result) == len(sample_dataframe), "Keep strategy should preserve row count"

    @pytest.mark.skipif(not HAS_CLEANER, reason="Cleaner module not importable")
    def test_standardize_data_types_returns_dataframe(self, sample_dataframe):
        """standardize_data_types should return a DataFrame."""
        result = standardize_data_types(sample_dataframe)
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"

    @pytest.mark.skipif(not HAS_CLEANER, reason="Cleaner module not importable")
    def test_standardize_data_types_numeric_columns(self, sample_dataframe):
        """standardize_data_types should make element columns numeric."""
        result = standardize_data_types(sample_dataframe)
        for col in ['Au_ppb', 'Cu_ppm', 'Pb_ppm', 'Zn_ppm', 'As_ppm']:
            if col in result.columns:
                assert pd.api.types.is_numeric_dtype(result[col]), \
                    f"Column {col} should be numeric after standardization"


# ============================================================================
# ANALYZER FUNCTION TESTS
# ============================================================================

class TestAnalyzerFunctions:
    """Tests for statistical analysis functions."""

    @pytest.mark.skipif(not HAS_ANALYZER, reason="Analyzer module not importable")
    def test_calculate_statistics_returns_dataframe(self, sample_dataframe):
        """calculate_statistics should return a DataFrame."""
        result = calculate_statistics(sample_dataframe)
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"

    @pytest.mark.skipif(not HAS_ANALYZER, reason="Analyzer module not importable")
    def test_calculate_statistics_contains_descriptive_stats(self, sample_dataframe):
        """calculate_statistics should contain basic statistical metrics."""
        result = calculate_statistics(sample_dataframe)
        expected_stats = ['count', 'mean', 'std', 'min', 'max']
        for stat in expected_stats:
            assert stat in result.index, f"Missing statistic '{stat}'"

    @pytest.mark.skipif(not HAS_ANALYZER, reason="Analyzer module not importable")
    def test_calculate_statistics_has_elements(self, sample_dataframe):
        """calculate_statistics should have element columns."""
        result = calculate_statistics(sample_dataframe)
        expected_elements = ['Au_ppb', 'Cu_ppm', 'Pb_ppm', 'Zn_ppm', 'As_ppm']
        for elem in expected_elements:
            assert elem in result.columns, f"Missing element '{elem}' in statistics"

    @pytest.mark.skipif(not HAS_ANALYZER, reason="Analyzer module not importable")
    def test_group_statistics_returns_dataframe(self, sample_dataframe):
        """group_statistics should return a DataFrame."""
        result = group_statistics(sample_dataframe, group_by='rock_type')
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"

    @pytest.mark.skipif(not HAS_ANALYZER, reason="Analyzer module not importable")
    def test_group_statistics_has_groups(self, sample_dataframe):
        """group_statistics should have group rows."""
        result = group_statistics(sample_dataframe, group_by='rock_type')
        expected_groups = sample_dataframe['rock_type'].unique()
        for group in expected_groups:
            assert group in result.index.get_level_values(0) or group in result.index, \
                   f"Missing group '{group}' in results"

    @pytest.mark.skipif(not HAS_ANALYZER, reason="Analyzer module not importable")
    def test_correlation_analysis_returns_dataframe(self, sample_dataframe):
        """correlation_analysis should return a DataFrame."""
        result = correlation_analysis(sample_dataframe)
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"

    @pytest.mark.skipif(not HAS_ANALYZER, reason="Analyzer module not importable")
    def test_correlation_analysis_is_square_matrix(self, sample_dataframe):
        """correlation_analysis should return square matrix."""
        result = correlation_analysis(sample_dataframe)
        assert result.shape[0] == result.shape[1], "Should return square matrix"

    @pytest.mark.skipif(not HAS_ANALYZER, reason="Analyzer module not importable")
    def test_correlation_analysis_diagonal_is_one(self, sample_dataframe):
        """correlation_analysis diagonal should be 1 (self-correlation)."""
        result = correlation_analysis(sample_dataframe)
        np.testing.assert_array_almost_equal(
            np.diag(result),
            np.ones(result.shape[0]),
            decimal=5,
            err_msg="Diagonal should be all 1s"
        )

    @pytest.mark.skipif(not HAS_ANALYZER, reason="Analyzer module not importable")
    def test_identify_significant_correlations_returns_list(self, sample_dataframe):
        """identify_significant_correlations should return list."""
        result = identify_significant_correlations(sample_dataframe)
        assert isinstance(result, list), "Should return list"

    @pytest.mark.skipif(not HAS_ANALYZER, reason="Analyzer module not importable")
    def test_identify_significant_correlations_dict_entries(self, sample_dataframe):
        """identify_significant_correlations should contain dict entries."""
        result = identify_significant_correlations(sample_dataframe, threshold=0.1)
        if len(result) > 0:
            expected_keys = ['element_1', 'element_2', 'correlation']
            for entry in result:
                assert isinstance(entry, dict), "Entries should be dicts"
                for key in expected_keys:
                    assert key in entry, f"Missing key '{key}' in correlation entry"

    @pytest.mark.skipif(not HAS_ANALYZER, reason="Analyzer module not importable")
    def test_analyze_by_spatial_region_returns_dict(self, sample_dataframe):
        """analyze_by_spatial_region should return a dict."""
        result = analyze_by_spatial_region(sample_dataframe, n_regions=2)
        assert isinstance(result, dict), "Should return dict"

    @pytest.mark.skipif(not HAS_ANALYZER, reason="Analyzer module not importable")
    def test_analyze_by_spatial_region_has_regions(self, sample_dataframe):
        """analyze_by_spatial_region should contain region data."""
        result = analyze_by_spatial_region(sample_dataframe, n_regions=2)
        assert len(result) > 0, "Should contain at least one region"

    @pytest.mark.skipif(not HAS_ANALYZER, reason="Analyzer module not importable")
    def test_generate_analysis_summary_returns_dict(self, sample_dataframe):
        """generate_analysis_summary should return a dict."""
        result = generate_analysis_summary(sample_dataframe)
        assert isinstance(result, dict), "Should return dict"

    @pytest.mark.skipif(not HAS_ANALYZER, reason="Analyzer module not importable")
    def test_generate_analysis_summary_has_required_keys(self, sample_dataframe):
        """generate_analysis_summary should have expected keys."""
        result = generate_analysis_summary(sample_dataframe)
        expected_keys = ['descriptive_stats', 'correlations']
        for key in expected_keys:
            assert key in result, f"Missing key '{key}' in analysis summary"


# ============================================================================
# DETECTOR FUNCTION TESTS
# ============================================================================

class TestDetectorFunctions:
    """Tests for anomaly detection functions."""

    @pytest.mark.skipif(not HAS_DETECTOR, reason="Detector module not importable")
    def test_detect_anomalies_returns_dataframe(self, sample_dataframe):
        """detect_anomalies should return a DataFrame."""
        result = detect_anomalies(sample_dataframe, percentile=95)
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"

    @pytest.mark.skipif(not HAS_DETECTOR, reason="Detector module not importable")
    def test_detect_anomalies_has_required_columns(self, sample_dataframe):
        """detect_anomalies result should have required columns."""
        result = detect_anomalies(sample_dataframe, percentile=95)
        expected_cols = ['sample_id', 'element', 'value', 'threshold']
        for col in expected_cols:
            assert col in result.columns or col in result.index.names, \
                   f"Missing column '{col}' in anomalies"

    @pytest.mark.skipif(not HAS_DETECTOR, reason="Detector module not importable")
    def test_detect_anomalies_finite_values(self, sample_dataframe):
        """detect_anomalies should return finite anomaly values."""
        result = detect_anomalies(sample_dataframe, percentile=95)
        if len(result) > 0:
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                assert np.all(np.isfinite(result[col])), \
                       f"Non-finite values in column '{col}'"

    @pytest.mark.skipif(not HAS_DETECTOR, reason="Detector module not importable")
    def test_calculate_thresholds_returns_dict(self, sample_dataframe):
        """calculate_thresholds should return a dictionary."""
        result = calculate_thresholds(sample_dataframe, percentile=95)
        assert isinstance(result, dict), "Should return dictionary"

    @pytest.mark.skipif(not HAS_DETECTOR, reason="Detector module not importable")
    def test_calculate_thresholds_has_elements(self, sample_dataframe):
        """calculate_thresholds should have element keys."""
        result = calculate_thresholds(sample_dataframe, percentile=95)
        expected_elements = ['Au_ppb', 'Cu_ppm', 'Pb_ppm', 'Zn_ppm', 'As_ppm']
        for elem in expected_elements:
            if elem in sample_dataframe.columns:
                assert elem in result, f"Missing threshold for '{elem}'"

    @pytest.mark.skipif(not HAS_DETECTOR, reason="Detector module not importable")
    def test_anomaly_detector_creation(self):
        """AnomalyDetector should be instantiable."""
        det = AnomalyDetector(percentile=95)
        assert det is not None
        assert det.percentile == 95

    @pytest.mark.skipif(not HAS_DETECTOR, reason="Detector module not importable")
    def test_anomaly_detector_fit(self, sample_dataframe):
        """AnomalyDetector.fit should work."""
        det = AnomalyDetector(percentile=95)
        result = det.fit(sample_dataframe)
        assert result is det, "fit() should return self"

    @pytest.mark.skipif(not HAS_DETECTOR, reason="Detector module not importable")
    def test_anomaly_detector_detect_after_fit(self, sample_dataframe):
        """AnomalyDetector.detect should work after fit."""
        det = AnomalyDetector(percentile=95)
        det.fit(sample_dataframe)
        result = det.detect(sample_dataframe)
        assert isinstance(result, pd.DataFrame), "detect() should return DataFrame"

    @pytest.mark.skipif(not HAS_DETECTOR, reason="Detector module not importable")
    def test_anomaly_detector_detect_before_fit_raises(self, sample_dataframe):
        """AnomalyDetector.detect should raise ValueError if not fitted."""
        det = AnomalyDetector(percentile=95)
        with pytest.raises(ValueError):
            det.detect(sample_dataframe)

    @pytest.mark.skipif(not HAS_DETECTOR, reason="Detector module not importable")
    def test_anomaly_detector_fit_detect(self, sample_dataframe):
        """AnomalyDetector.fit_detect should work."""
        det = AnomalyDetector(percentile=95)
        result = det.fit_detect(sample_dataframe)
        assert isinstance(result, pd.DataFrame), "fit_detect() should return DataFrame"

    @pytest.mark.skipif(not HAS_DETECTOR, reason="Detector module not importable")
    def test_anomaly_detector_get_thresholds_after_fit(self, sample_dataframe):
        """AnomalyDetector.get_thresholds should work after fit."""
        det = AnomalyDetector(percentile=95)
        det.fit(sample_dataframe)
        thresholds = det.get_thresholds()
        assert isinstance(thresholds, dict), "Should return dict"
        assert len(thresholds) > 0, "Should have thresholds"

    @pytest.mark.skipif(not HAS_DETECTOR, reason="Detector module not importable")
    def test_get_anomaly_spatial_context_returns_dataframe(self, sample_dataframe):
        """get_anomaly_spatial_context should return a DataFrame."""
        anomalies = detect_anomalies(sample_dataframe, percentile=90)
        if len(anomalies) > 0:
            result = get_anomaly_spatial_context(anomalies, sample_dataframe)
            assert isinstance(result, pd.DataFrame), "Should return DataFrame"

    @pytest.mark.skipif(not HAS_DETECTOR, reason="Detector module not importable")
    def test_summarize_anomalies_returns_dict(self, sample_dataframe):
        """summarize_anomalies should return a dict."""
        anomalies = detect_anomalies(sample_dataframe, percentile=90)
        result = summarize_anomalies(anomalies, sample_dataframe)
        assert isinstance(result, dict), "Should return dict"

    @pytest.mark.skipif(not HAS_DETECTOR, reason="Detector module not importable")
    def test_summarize_anomalies_has_total(self, sample_dataframe):
        """summarize_anomalies should include total_anomalies key."""
        anomalies = detect_anomalies(sample_dataframe, percentile=90)
        result = summarize_anomalies(anomalies, sample_dataframe)
        assert 'total_anomalies' in result, "Should have 'total_anomalies' key"


# ============================================================================
# REPORTER FUNCTION TESTS
# ============================================================================

class TestReporterFunctions:
    """Tests for report generation functions."""

    @pytest.mark.skipif(not HAS_REPORTER, reason="Reporter module not importable")
    def test_generate_text_report_accepts_params(self, sample_dataframe):
        """generate_text_report should accept required parameters."""
        analysis_results = {'mean': 5.0, 'std': 1.5}
        anomalies = pd.DataFrame({'sample_id': ['S001'], 'element': ['Au_ppb']})
        variant_config = {'elements': ['Au_ppb']}

        # Should not raise an error
        result = generate_text_report(analysis_results, anomalies, variant_config)
        # Result should be string or None, but not raise
        assert result is None or isinstance(result, str), \
               "generate_text_report should return str or None"

    @pytest.mark.skipif(not HAS_REPORTER, reason="Reporter module not importable")
    def test_save_report_accepts_params(self, tmp_path):
        """save_report should accept required parameters."""
        report_content = "Test Report"
        output_path = tmp_path / "test_report.txt"

        # Should not raise an error
        result = save_report(str(report_content), str(output_path))
        # Result should be bool or None, but not raise
        assert result is None or isinstance(result, bool), \
               "save_report should return bool or None"

    @pytest.mark.skipif(not HAS_REPORTER, reason="Reporter module not importable")
    def test_generate_html_report_returns_string(self):
        """generate_html_report should return a string."""
        analysis_results = {'mean': 5.0, 'std': 1.5}
        anomalies = pd.DataFrame({'sample_id': ['S001'], 'element': ['Au_ppb']})
        figures = []
        variant_config = {'elements': ['Au_ppb']}

        result = generate_html_report(analysis_results, anomalies, figures, variant_config)
        assert result is None or isinstance(result, str), \
               "generate_html_report should return str or None"

    @pytest.mark.skipif(not HAS_REPORTER, reason="Reporter module not importable")
    def test_create_executive_summary_returns_string(self):
        """create_executive_summary should return a string."""
        analysis_results = {'mean': 5.0, 'std': 1.5}
        anomalies = pd.DataFrame({'sample_id': ['S001'], 'element': ['Au_ppb']})
        variant_config = {'elements': ['Au_ppb']}

        result = create_executive_summary(analysis_results, anomalies, variant_config)
        assert result is None or isinstance(result, str), \
               "create_executive_summary should return str or None"


# ============================================================================
# VISUALIZER FUNCTION TESTS
# ============================================================================

class TestVisualizerFunctions:
    """Tests for visualization generation functions."""

    @pytest.fixture(autouse=True)
    def cleanup_matplotlib(self):
        """Clean up matplotlib figures after each test."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        yield
        plt.close('all')

    @pytest.mark.skipif(not HAS_VISUALIZER, reason="Visualizer module not importable")
    def test_apply_professional_style_no_error(self):
        """apply_professional_style should not raise an error."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        # Should not raise
        apply_professional_style(ax, title="Test", xlabel="X", ylabel="Y")

    @pytest.mark.skipif(not HAS_VISUALIZER, reason="Visualizer module not importable")
    def test_plot_element_histograms_creates_figure(self, sample_dataframe, tmp_path):
        """plot_element_histograms should create a figure."""
        fig = plot_element_histograms(
            sample_dataframe,
            elements=['Au_ppb', 'Cu_ppm']
        )
        assert fig is not None, "Should return a figure"
        output_path = tmp_path / "histograms.png"
        fig.savefig(str(output_path))
        assert output_path.exists(), "Should save histogram file"
        assert output_path.stat().st_size > 0, "Saved file should not be empty"

    @pytest.mark.skipif(not HAS_VISUALIZER, reason="Visualizer module not importable")
    def test_plot_spatial_distribution_creates_figure(self, sample_dataframe, tmp_path):
        """plot_spatial_distribution should create a figure."""
        fig = plot_spatial_distribution(sample_dataframe, "Au_ppb")
        assert fig is not None, "Should return a figure"
        output_path = tmp_path / "spatial.png"
        fig.savefig(str(output_path))
        assert output_path.exists(), "Should save spatial plot file"
        assert output_path.stat().st_size > 0, "Saved file should not be empty"

    @pytest.mark.skipif(not HAS_VISUALIZER, reason="Visualizer module not importable")
    def test_plot_anomaly_map_creates_figure(self, sample_dataframe, tmp_path):
        """plot_anomaly_map should create a figure."""
        anomalies = pd.DataFrame({
            'sample_id': ['S001'],
            'utm_east': [505000.0],
            'utm_north': [8205000.0],
            'element': ['Au_ppb'],
            'value': [100.0],
            'threshold': [50.0]
        })
        fig = plot_anomaly_map(sample_dataframe, anomalies, "Au_ppb")
        assert fig is not None, "Should return a figure"
        output_path = tmp_path / "anomaly_map.png"
        fig.savefig(str(output_path))
        assert output_path.exists(), "Should save anomaly map file"

    @pytest.mark.skipif(not HAS_VISUALIZER, reason="Visualizer module not importable")
    def test_plot_correlation_heatmap_creates_figure(self, sample_dataframe, tmp_path):
        """plot_correlation_heatmap should create a figure."""
        elements = ['Au_ppb', 'Cu_ppm', 'Pb_ppm', 'Zn_ppm', 'As_ppm']
        present = [e for e in elements if e in sample_dataframe.columns]
        corr_matrix = sample_dataframe[present].corr()
        fig = plot_correlation_heatmap(corr_matrix)
        assert fig is not None, "Should return a figure"
        output_path = tmp_path / "heatmap.png"
        fig.savefig(str(output_path))
        assert output_path.exists(), "Should save heatmap file"
        assert output_path.stat().st_size > 0, "Saved file should not be empty"

    @pytest.mark.skipif(not HAS_VISUALIZER, reason="Visualizer module not importable")
    def test_create_summary_plot_creates_figure(self, sample_dataframe, tmp_path):
        """create_summary_plot should create a figure."""
        stats = sample_dataframe[['Au_ppb', 'Cu_ppm']].describe()
        anomalies = pd.DataFrame({
            'sample_id': ['S001'],
            'element': ['Au_ppb'],
            'value': [100.0],
            'threshold': [50.0]
        })
        fig = create_summary_plot(sample_dataframe, stats, anomalies,
                                  elements=['Au_ppb', 'Cu_ppm'])
        assert fig is not None, "Should return a figure"
        output_path = tmp_path / "summary.png"
        fig.savefig(str(output_path))
        assert output_path.exists(), "Should save summary plot file"

    @pytest.mark.skipif(not HAS_VISUALIZER, reason="Visualizer module not importable")
    def test_save_figure_creates_file(self, tmp_path):
        """save_figure should save a figure to file."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        output_path = tmp_path / "test_figure.png"
        save_figure(fig, str(output_path))
        assert output_path.exists(), "Should create output file"
        assert output_path.stat().st_size > 0, "Saved file should not be empty"


# ============================================================================
# MAIN PIPELINE TESTS
# ============================================================================

class TestMainPipeline:
    """Tests for main pipeline orchestration."""

    @pytest.mark.skipif(not HAS_MAIN, reason="Main module not importable")
    def test_run_pipeline_returns_dict_or_none(self):
        """run_pipeline should return a dict or None."""
        result = run_pipeline()
        assert result is None or isinstance(result, dict), \
               "run_pipeline should return dict or None"

    @pytest.mark.skipif(not HAS_MAIN, reason="Main module not importable")
    def test_run_pipeline_has_expected_keys(self):
        """run_pipeline result should have expected keys if implemented."""
        result = run_pipeline()
        if result is not None:
            assert 'status' in result or 'anomaly_count' in result, \
                   "Pipeline result should contain status or anomaly_count"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests across multiple modules."""

    @pytest.mark.skipif(not (HAS_LOADER and HAS_CLEANER and HAS_ANALYZER),
                        reason="Not all modules importable")
    def test_pipeline_workflow_load_clean_analyze(self, sample_csv):
        """Test basic pipeline workflow: load -> clean -> analyze."""
        csv_path, _ = sample_csv

        # Load
        df = load_samples(csv_path)
        assert df is not None and len(df) > 0

        # Clean
        cleaned_df, _ = clean_data(df)
        assert cleaned_df is not None

        # Analyze
        stats = calculate_statistics(cleaned_df)
        assert stats is not None

    @pytest.mark.skipif(not (HAS_LOADER and HAS_DETECTOR),
                        reason="Not all modules importable")
    def test_pipeline_workflow_with_anomaly_detection(self, sample_csv):
        """Test pipeline with anomaly detection."""
        csv_path, _ = sample_csv

        # Load
        df = load_samples(csv_path)
        assert df is not None

        # Detect anomalies
        anomalies = detect_anomalies(df, percentile=90)
        assert anomalies is not None
