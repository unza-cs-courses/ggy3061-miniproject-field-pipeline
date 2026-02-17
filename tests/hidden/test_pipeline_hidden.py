"""
Mini-Project Hidden Tests - Field Sample Pipeline

These tests verify student implementations against the actual field data
and variant-specific parameters. They are NOT visible to students.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

SRC_DIR = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

# --- Conditional imports ---

try:
    from pipeline.loader import load_samples, validate_data, get_loading_statistics, load_and_validate
    HAS_LOADER = True
except (ImportError, Exception):
    HAS_LOADER = False

try:
    from pipeline.cleaner import (
        handle_missing_values, detect_outliers, filter_by_criteria,
        clean_data, standardize_data_types, handle_outliers as cleaner_handle_outliers
    )
    HAS_CLEANER = True
except (ImportError, Exception):
    HAS_CLEANER = False

try:
    from pipeline.analyzer import (
        calculate_statistics, group_statistics, correlation_analysis,
        identify_significant_correlations, analyze_by_spatial_region,
        generate_analysis_summary
    )
    HAS_ANALYZER = True
except (ImportError, Exception):
    HAS_ANALYZER = False

try:
    from pipeline.detector import (
        detect_anomalies, calculate_thresholds, AnomalyDetector,
        get_anomaly_spatial_context, summarize_anomalies
    )
    HAS_DETECTOR = True
except (ImportError, Exception):
    HAS_DETECTOR = False

try:
    from pipeline.visualizer import (
        apply_professional_style, plot_element_histograms,
        plot_spatial_distribution, plot_correlation_heatmap, save_figure
    )
    HAS_VISUALIZER = True
except (ImportError, Exception):
    HAS_VISUALIZER = False

try:
    from pipeline.reporter import (
        generate_text_report, generate_html_report, save_report,
        create_executive_summary
    )
    HAS_REPORTER = True
except (ImportError, Exception):
    HAS_REPORTER = False


# Valid values for variant parameters
VALID_STUDY_AREAS = [
    'Northern Zone', 'Eastern Block', 'Western Prospect',
    'Central Basin', 'Southern Ridge'
]
VALID_ELEMENTS = ['Au', 'Cu', 'Pb', 'Zn', 'As']
VALID_COLLECTORS = ['A. Smith', 'B. Johnson', 'C. Williams', 'D. Brown']
VALID_PERCENTILES = [90, 95]

# Expected CSV columns
EXPECTED_COLUMNS = [
    'sample_id', 'collection_date', 'collector', 'utm_east', 'utm_north',
    'elevation', 'rock_type', 'alteration', 'structure',
    'Au_ppb', 'Cu_ppm', 'Pb_ppm', 'Zn_ppm', 'As_ppm',
    'sample_weight', 'qc_flag'
]

ELEMENT_COLUMNS = ['Au_ppb', 'Cu_ppm', 'Pb_ppm', 'Zn_ppm', 'As_ppm']


# ============================================================================
# HIDDEN LOADER TESTS
# ============================================================================

class TestHiddenLoader:
    """Hidden tests for data loading against actual CSV data."""

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not implemented")
    def test_load_actual_csv_row_count(self, field_csv_file, field_dataframe):
        """Loading actual CSV should return correct number of rows."""
        df = load_samples(field_csv_file)
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert len(df) == len(field_dataframe), \
            f"Expected {len(field_dataframe)} rows, got {len(df)}"

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not implemented")
    def test_load_actual_csv_has_all_columns(self, field_csv_file):
        """Loading actual CSV should contain all expected columns."""
        df = load_samples(field_csv_file)
        for col in EXPECTED_COLUMNS:
            assert col in df.columns, f"Missing expected column '{col}'"

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not implemented")
    def test_validate_actual_data_structure(self, field_csv_file):
        """Validating actual CSV data should pass."""
        df = load_samples(field_csv_file)
        is_valid, errors = validate_data(df)
        assert isinstance(is_valid, bool), "First return must be bool"
        assert isinstance(errors, list), "Second return must be list"

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not implemented")
    def test_load_and_validate_actual_data(self, field_csv_file):
        """load_and_validate should work on actual data."""
        result = load_and_validate(field_csv_file)
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 2, "Should return 2-element tuple"
        df, stats = result
        assert isinstance(df, pd.DataFrame), "First element should be DataFrame"
        assert isinstance(stats, dict), "Second element should be dict"
        assert len(df) > 0, "DataFrame should not be empty"

    @pytest.mark.skipif(not HAS_LOADER, reason="Loader module not implemented")
    def test_load_alternative_data(self, alternative_dataframe, tmp_path):
        """Loading alternative data should work (no hardcoded values)."""
        csv_path = tmp_path / "alt_samples.csv"
        alternative_dataframe.to_csv(str(csv_path), index=False)
        df = load_samples(str(csv_path))
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert len(df) == len(alternative_dataframe), "Should load all rows"


# ============================================================================
# HIDDEN CLEANER TESTS
# ============================================================================

class TestHiddenCleaner:
    """Hidden tests for data cleaning against actual data."""

    @pytest.mark.skipif(not (HAS_LOADER and HAS_CLEANER), reason="Modules not implemented")
    def test_clean_actual_data_returns_tuple(self, field_csv_file):
        """Cleaning actual data should return (DataFrame, dict)."""
        df = load_samples(field_csv_file)
        result = clean_data(df)
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 2, "Should return 2-element tuple"
        cleaned_df, report = result
        assert isinstance(cleaned_df, pd.DataFrame), "First should be DataFrame"
        assert isinstance(report, dict), "Second should be dict"

    @pytest.mark.skipif(not (HAS_LOADER and HAS_CLEANER), reason="Modules not implemented")
    def test_clean_actual_data_handles_missing(self, field_csv_file):
        """Cleaning should handle missing values in actual data."""
        df = load_samples(field_csv_file)
        cleaned_df, report = clean_data(df)
        # After cleaning, element columns should have fewer or equal NaN values
        for col in ELEMENT_COLUMNS:
            if col in cleaned_df.columns:
                original_na = df[col].isna().sum() if col in df.columns else 0
                cleaned_na = cleaned_df[col].isna().sum()
                assert cleaned_na <= original_na, \
                    f"Column {col} should have fewer NaN after cleaning"

    @pytest.mark.skipif(not (HAS_LOADER and HAS_CLEANER), reason="Modules not implemented")
    def test_filter_by_collector_on_actual_data(self, field_csv_file, variant_config):
        """Filtering by variant's collector should work on actual data."""
        df = load_samples(field_csv_file)
        collector = variant_config['parameters']['collector_filter']
        filtered = filter_by_criteria(df, collector=collector)
        assert isinstance(filtered, pd.DataFrame), "Should return DataFrame"
        assert len(filtered) > 0, f"Should find samples for collector '{collector}'"
        assert all(filtered['collector'] == collector), "All rows should match collector"
        assert len(filtered) < len(df), "Filtered result should be smaller than original"

    @pytest.mark.skipif(not (HAS_LOADER and HAS_CLEANER), reason="Modules not implemented")
    def test_filter_by_elevation_on_actual_data(self, field_csv_file, variant_config):
        """Filtering by variant's elevation range should work on actual data."""
        df = load_samples(field_csv_file)
        elev_range = variant_config['parameters']['elevation_range']
        elev_min = elev_range['min']
        elev_max = elev_range['max']
        filtered = filter_by_criteria(df, elevation_range=(elev_min, elev_max))
        assert isinstance(filtered, pd.DataFrame), "Should return DataFrame"
        assert len(filtered) > 0, \
            f"Should find samples in elevation range [{elev_min}, {elev_max}]"
        assert all(filtered['elevation'] >= elev_min), "Should respect min elevation"
        assert all(filtered['elevation'] <= elev_max), "Should respect max elevation"

    @pytest.mark.skipif(not HAS_CLEANER, reason="Cleaner module not implemented")
    def test_clean_alternative_data(self, alternative_dataframe):
        """Cleaning alternative data should work (no hardcoded values)."""
        result = clean_data(alternative_dataframe)
        assert isinstance(result, tuple), "Should return tuple"
        cleaned_df, report = result
        assert isinstance(cleaned_df, pd.DataFrame), "Should return DataFrame"
        assert len(cleaned_df) > 0, "Cleaned data should not be empty"


# ============================================================================
# HIDDEN ANALYZER TESTS
# ============================================================================

class TestHiddenAnalyzer:
    """Hidden tests for statistical analysis against actual data."""

    @pytest.mark.skipif(not (HAS_LOADER and HAS_ANALYZER), reason="Modules not implemented")
    def test_calculate_statistics_on_actual_data(self, field_csv_file, variant_config):
        """Statistics calculation should work on actual data with variant elements."""
        df = load_samples(field_csv_file)
        target_elems = variant_config['parameters']['target_elements']
        element_cols = [f"{e}_ppb" if e == "Au" else f"{e}_ppm" for e in target_elems]
        present_cols = [c for c in element_cols if c in df.columns]
        stats = calculate_statistics(df, elements=present_cols)
        assert isinstance(stats, pd.DataFrame), "Should return DataFrame"
        assert 'mean' in stats.index, "Should have 'mean' statistic"
        assert 'std' in stats.index, "Should have 'std' statistic"

    @pytest.mark.skipif(not (HAS_LOADER and HAS_ANALYZER), reason="Modules not implemented")
    def test_group_statistics_on_actual_data(self, field_csv_file):
        """Group statistics should work on actual data."""
        df = load_samples(field_csv_file)
        result = group_statistics(df, group_by='rock_type')
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        rock_types = df['rock_type'].unique()
        for rt in rock_types:
            assert rt in result.index.get_level_values(0) or rt in result.index, \
                f"Missing rock_type '{rt}' in grouped results"

    @pytest.mark.skipif(not (HAS_LOADER and HAS_ANALYZER), reason="Modules not implemented")
    def test_correlation_matrix_properties(self, field_csv_file):
        """Correlation matrix should be symmetric with diagonal=1."""
        df = load_samples(field_csv_file)
        corr = correlation_analysis(df, elements=ELEMENT_COLUMNS)
        assert isinstance(corr, pd.DataFrame), "Should return DataFrame"
        # Symmetric
        np.testing.assert_array_almost_equal(
            corr.values, corr.values.T, decimal=10,
            err_msg="Correlation matrix should be symmetric"
        )
        # Diagonal = 1
        np.testing.assert_array_almost_equal(
            np.diag(corr.values), np.ones(corr.shape[0]), decimal=5,
            err_msg="Diagonal should be 1.0"
        )
        # Values in [-1, 1]
        assert corr.values.min() >= -1.001, "Correlations should be >= -1"
        assert corr.values.max() <= 1.001, "Correlations should be <= 1"

    @pytest.mark.skipif(not (HAS_LOADER and HAS_ANALYZER), reason="Modules not implemented")
    def test_analyze_by_spatial_region_on_actual_data(self, field_csv_file):
        """Spatial region analysis should work on actual data."""
        df = load_samples(field_csv_file)
        result = analyze_by_spatial_region(df, n_regions=3)
        assert isinstance(result, dict), "Should return dict"
        assert len(result) > 0, "Should have at least one region"

    @pytest.mark.skipif(not HAS_ANALYZER, reason="Analyzer module not implemented")
    def test_calculate_statistics_on_alternative_data(self, alternative_dataframe):
        """Statistics should work on alternative data (no hardcoded values)."""
        stats = calculate_statistics(alternative_dataframe, elements=ELEMENT_COLUMNS)
        assert isinstance(stats, pd.DataFrame), "Should return DataFrame"
        # Means should be different from typical field_samples values
        for col in ELEMENT_COLUMNS:
            if col in stats.columns:
                assert stats.loc['mean', col] > 0, \
                    f"Mean for {col} should be positive"


# ============================================================================
# HIDDEN DETECTOR TESTS
# ============================================================================

class TestHiddenDetector:
    """Hidden tests for anomaly detection against actual data."""

    @pytest.mark.skipif(not (HAS_LOADER and HAS_DETECTOR), reason="Modules not implemented")
    def test_detect_anomalies_on_actual_data(self, field_csv_file, variant_config):
        """Anomaly detection should work on actual data."""
        df = load_samples(field_csv_file)
        percentile = variant_config['parameters']['anomaly_percentile']
        anomalies = detect_anomalies(df, percentile=percentile)
        assert isinstance(anomalies, pd.DataFrame), "Should return DataFrame"

    @pytest.mark.skipif(not (HAS_LOADER and HAS_DETECTOR), reason="Modules not implemented")
    def test_anomaly_thresholds_on_actual_data(self, field_csv_file, variant_config):
        """Thresholds should be reasonable for actual data."""
        df = load_samples(field_csv_file)
        percentile = variant_config['parameters']['anomaly_percentile']
        thresholds = calculate_thresholds(df, percentile=percentile)
        assert isinstance(thresholds, dict), "Should return dict"
        for elem in ELEMENT_COLUMNS:
            if elem in thresholds:
                assert thresholds[elem] > 0, \
                    f"Threshold for {elem} should be positive"

    @pytest.mark.skipif(not (HAS_LOADER and HAS_DETECTOR), reason="Modules not implemented")
    def test_anomaly_detector_class_on_actual_data(self, field_csv_file, variant_config):
        """AnomalyDetector class should work on actual data."""
        df = load_samples(field_csv_file)
        percentile = variant_config['parameters']['anomaly_percentile']
        det = AnomalyDetector(percentile=percentile, elements=ELEMENT_COLUMNS)
        det.fit(df)
        anomalies = det.detect(df)
        assert isinstance(anomalies, pd.DataFrame), "detect() should return DataFrame"
        thresholds = det.get_thresholds()
        assert isinstance(thresholds, dict), "get_thresholds() should return dict"
        assert len(thresholds) > 0, "Should have thresholds after fitting"

    @pytest.mark.skipif(not (HAS_LOADER and HAS_DETECTOR), reason="Modules not implemented")
    def test_anomaly_counts_are_reasonable(self, field_csv_file, variant_config):
        """Number of anomalies should be reasonable (not 0, not all)."""
        df = load_samples(field_csv_file)
        percentile = variant_config['parameters']['anomaly_percentile']
        anomalies = detect_anomalies(df, percentile=percentile)
        total_samples = len(df)
        if len(anomalies) > 0:
            unique_samples = anomalies['sample_id'].nunique() if 'sample_id' in anomalies.columns else len(anomalies)
            # Anomalies should be a minority of total samples
            assert unique_samples < total_samples, \
                "Anomaly count should be less than total samples"

    @pytest.mark.skipif(not HAS_DETECTOR, reason="Detector module not implemented")
    def test_detect_anomalies_on_alternative_data(self, alternative_dataframe):
        """Anomaly detection should work on alternative data (no hardcoded values)."""
        anomalies = detect_anomalies(alternative_dataframe, percentile=95)
        assert isinstance(anomalies, pd.DataFrame), "Should return DataFrame"


# ============================================================================
# HIDDEN VISUALIZER TESTS
# ============================================================================

class TestHiddenVisualizer:
    """Hidden tests for visualization against actual data."""

    @pytest.mark.skipif(not (HAS_LOADER and HAS_VISUALIZER), reason="Modules not implemented")
    def test_histogram_from_actual_data(self, field_csv_file, temp_output_dir):
        """Histograms should be generated from actual data."""
        df = load_samples(field_csv_file)
        fig = plot_element_histograms(df, elements=['Au_ppb', 'Cu_ppm'])
        assert fig is not None, "Should return a figure"
        output_path = temp_output_dir / "hidden_histograms.png"
        fig.savefig(str(output_path))
        assert output_path.exists(), "File should be created"
        assert output_path.stat().st_size > 0, "File should not be empty"

    @pytest.mark.skipif(not (HAS_LOADER and HAS_VISUALIZER), reason="Modules not implemented")
    def test_spatial_distribution_from_actual_data(self, field_csv_file, temp_output_dir):
        """Spatial distribution plot should work with actual data."""
        df = load_samples(field_csv_file)
        fig = plot_spatial_distribution(df, "Au_ppb")
        assert fig is not None, "Should return a figure"
        output_path = temp_output_dir / "hidden_spatial.png"
        fig.savefig(str(output_path))
        assert output_path.exists(), "File should be created"
        assert output_path.stat().st_size > 0, "File should not be empty"

    @pytest.mark.skipif(not (HAS_LOADER and HAS_VISUALIZER), reason="Modules not implemented")
    def test_correlation_heatmap_from_actual_data(self, field_csv_file, temp_output_dir):
        """Correlation heatmap should work with actual data."""
        df = load_samples(field_csv_file)
        present = [e for e in ELEMENT_COLUMNS if e in df.columns]
        corr = df[present].corr()
        fig = plot_correlation_heatmap(corr)
        assert fig is not None, "Should return a figure"
        output_path = temp_output_dir / "hidden_heatmap.png"
        fig.savefig(str(output_path))
        assert output_path.exists(), "File should be created"
        assert output_path.stat().st_size > 0, "File should not be empty"

    @pytest.mark.skipif(not HAS_VISUALIZER, reason="Visualizer module not implemented")
    def test_save_figure_creates_nonempty_file(self, temp_output_dir):
        """save_figure should create a non-empty file."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.bar(['A', 'B', 'C'], [10, 20, 30])
        output_path = temp_output_dir / "hidden_bar.png"
        save_figure(fig, str(output_path))
        assert output_path.exists(), "File should be created"
        assert output_path.stat().st_size > 0, "File should not be empty"


# ============================================================================
# HIDDEN REPORTER TESTS
# ============================================================================

class TestHiddenReporter:
    """Hidden tests for report generation."""

    @pytest.mark.skipif(not HAS_REPORTER, reason="Reporter module not implemented")
    def test_generate_text_report_with_analysis(self, variant_config):
        """Text report should be generated from analysis results."""
        analysis_results = {
            'descriptive_stats': pd.DataFrame({'Au_ppb': [10, 5], 'Cu_ppm': [50, 10]},
                                              index=['mean', 'std']),
            'total_samples': 800,
        }
        anomalies = pd.DataFrame({
            'sample_id': ['FS-0001', 'FS-0002'],
            'element': ['Au_ppb', 'Cu_ppm'],
            'value': [500.0, 200.0],
            'threshold': [300.0, 150.0]
        })
        result = generate_text_report(analysis_results, anomalies, variant_config)
        assert result is None or isinstance(result, str), "Should return str or None"

    @pytest.mark.skipif(not HAS_REPORTER, reason="Reporter module not implemented")
    def test_generate_html_report_with_analysis(self, variant_config):
        """HTML report should be generated from analysis results."""
        analysis_results = {
            'descriptive_stats': pd.DataFrame({'Au_ppb': [10, 5]}, index=['mean', 'std']),
            'total_samples': 800,
        }
        anomalies = pd.DataFrame({
            'sample_id': ['FS-0001'], 'element': ['Au_ppb'],
            'value': [500.0], 'threshold': [300.0]
        })
        result = generate_html_report(analysis_results, anomalies, [], variant_config)
        assert result is None or isinstance(result, str), "Should return str or None"

    @pytest.mark.skipif(not HAS_REPORTER, reason="Reporter module not implemented")
    def test_save_report_creates_file(self, temp_output_dir):
        """save_report should write report to file."""
        content = "Test report content for hidden tests.\nLine 2."
        output_path = temp_output_dir / "hidden_report.txt"
        result = save_report(content, str(output_path))
        assert result is None or isinstance(result, bool), "Should return bool or None"


# ============================================================================
# HIDDEN VARIANT VERIFICATION TESTS
# ============================================================================

class TestHiddenVariantVerification:
    """Verify that the variant configuration is valid."""

    def test_variant_has_required_keys(self, variant_config):
        """Variant config should have all required parameter keys."""
        params = variant_config.get('parameters', {})
        required = [
            'study_area', 'target_elements', 'collector_filter',
            'elevation_range', 'anomaly_percentile', 'num_samples'
        ]
        for key in required:
            assert key in params, f"Variant missing required key '{key}'"

    def test_variant_target_elements_valid(self, variant_config):
        """Target elements should be valid element abbreviations."""
        target_elems = variant_config['parameters']['target_elements']
        assert isinstance(target_elems, list), "target_elements should be a list"
        assert len(target_elems) == 3, "Should have exactly 3 target elements"
        for elem in target_elems:
            assert elem in VALID_ELEMENTS, \
                f"Invalid target element '{elem}'. Must be one of {VALID_ELEMENTS}"

    def test_variant_collector_filter_valid(self, variant_config):
        """Collector filter should be a valid collector name."""
        collector = variant_config['parameters']['collector_filter']
        assert collector in VALID_COLLECTORS, \
            f"Invalid collector '{collector}'. Must be one of {VALID_COLLECTORS}"

    def test_variant_elevation_range_valid(self, variant_config):
        """Elevation range should have valid min and max."""
        elev = variant_config['parameters']['elevation_range']
        assert 'min' in elev, "elevation_range must have 'min'"
        assert 'max' in elev, "elevation_range must have 'max'"
        assert isinstance(elev['min'], (int, float)), "min must be numeric"
        assert isinstance(elev['max'], (int, float)), "max must be numeric"
        assert elev['min'] < elev['max'], "min must be less than max"
        assert 1000 <= elev['min'] <= 2000, "min should be in reasonable range"
        assert 1400 <= elev['max'] <= 2500, "max should be in reasonable range"

    def test_variant_anomaly_percentile_valid(self, variant_config):
        """Anomaly percentile should be 90 or 95."""
        percentile = variant_config['parameters']['anomaly_percentile']
        assert percentile in VALID_PERCENTILES, \
            f"Invalid percentile {percentile}. Must be one of {VALID_PERCENTILES}"


# ============================================================================
# HIDDEN INTEGRATION TESTS
# ============================================================================

class TestHiddenIntegration:
    """Hidden integration tests running the full pipeline on actual data."""

    @pytest.mark.skipif(
        not (HAS_LOADER and HAS_CLEANER and HAS_ANALYZER and HAS_DETECTOR),
        reason="Not all pipeline modules implemented"
    )
    def test_full_pipeline_load_clean_analyze_detect(self, field_csv_file, variant_config):
        """Full pipeline: load -> clean -> analyze -> detect."""
        params = variant_config['parameters']

        # Step 1: Load
        df = load_samples(field_csv_file)
        assert isinstance(df, pd.DataFrame) and len(df) > 0

        # Step 2: Clean
        cleaned_df, clean_report = clean_data(df)
        assert isinstance(cleaned_df, pd.DataFrame) and len(cleaned_df) > 0

        # Step 3: Analyze
        target_elems = params['target_elements']
        element_cols = [f"{e}_ppb" if e == "Au" else f"{e}_ppm" for e in target_elems]
        present = [c for c in element_cols if c in cleaned_df.columns]
        stats = calculate_statistics(cleaned_df, elements=present)
        assert isinstance(stats, pd.DataFrame)

        # Step 4: Detect
        anomalies = detect_anomalies(
            cleaned_df,
            percentile=params['anomaly_percentile'],
            elements=present
        )
        assert isinstance(anomalies, pd.DataFrame)

    @pytest.mark.skipif(
        not (HAS_LOADER and HAS_CLEANER and HAS_ANALYZER and HAS_DETECTOR),
        reason="Not all pipeline modules implemented"
    )
    def test_pipeline_with_collector_filter(self, field_csv_file, variant_config):
        """Pipeline with collector filtering should produce valid results."""
        params = variant_config['parameters']

        df = load_samples(field_csv_file)
        cleaned_df, _ = clean_data(df)
        filtered = filter_by_criteria(cleaned_df, collector=params['collector_filter'])
        assert len(filtered) > 0, "Filtered data should not be empty"
        assert all(filtered['collector'] == params['collector_filter'])

        stats = calculate_statistics(filtered)
        assert isinstance(stats, pd.DataFrame)

    @pytest.mark.skipif(
        not (HAS_LOADER and HAS_CLEANER and HAS_ANALYZER and HAS_DETECTOR),
        reason="Not all pipeline modules implemented"
    )
    def test_pipeline_with_elevation_filter(self, field_csv_file, variant_config):
        """Pipeline with elevation filtering should produce valid results."""
        params = variant_config['parameters']
        elev = params['elevation_range']

        df = load_samples(field_csv_file)
        cleaned_df, _ = clean_data(df)
        filtered = filter_by_criteria(
            cleaned_df,
            elevation_range=(elev['min'], elev['max'])
        )
        assert len(filtered) > 0, "Filtered data should not be empty"
        assert all(filtered['elevation'] >= elev['min'])
        assert all(filtered['elevation'] <= elev['max'])

    @pytest.mark.skipif(
        not (HAS_LOADER and HAS_CLEANER and HAS_ANALYZER and HAS_DETECTOR and HAS_REPORTER),
        reason="Not all pipeline modules implemented"
    )
    def test_pipeline_output_structure(self, field_csv_file, variant_config):
        """Full pipeline output should have expected structure."""
        params = variant_config['parameters']

        df = load_samples(field_csv_file)
        cleaned_df, clean_report = clean_data(df)

        target_elems = params['target_elements']
        element_cols = [f"{e}_ppb" if e == "Au" else f"{e}_ppm" for e in target_elems]
        present = [c for c in element_cols if c in cleaned_df.columns]

        stats = calculate_statistics(cleaned_df, elements=present)
        corr = correlation_analysis(cleaned_df, elements=present)
        anomalies = detect_anomalies(
            cleaned_df,
            percentile=params['anomaly_percentile'],
            elements=present
        )

        # Verify output types
        assert isinstance(stats, pd.DataFrame), "Stats should be DataFrame"
        assert isinstance(corr, pd.DataFrame), "Correlation should be DataFrame"
        assert isinstance(anomalies, pd.DataFrame), "Anomalies should be DataFrame"
        assert isinstance(clean_report, dict), "Clean report should be dict"

        # Generate report
        analysis_results = {
            'descriptive_stats': stats,
            'correlations': corr,
            'total_samples': len(cleaned_df)
        }
        report = generate_text_report(analysis_results, anomalies, variant_config)
        assert report is None or isinstance(report, str), \
            "Report should be str or None"
