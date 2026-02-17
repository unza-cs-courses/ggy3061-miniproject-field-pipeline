"""
Data Cleaning and Preprocessing Module
======================================

This module handles data cleaning, missing value treatment,
outlier handling, and filtering based on quality criteria.

Functions:
    clean_data: Main cleaning function with configurable options
    handle_missing_values: Handle missing data with various strategies
    detect_outliers: Identify outliers using statistical methods
    filter_by_criteria: Filter data based on collector, elevation, etc.

Example:
    >>> from pipeline.cleaner import clean_data, filter_by_criteria
    >>> cleaned = clean_data(raw_df, strategy="median")
    >>> filtered = filter_by_criteria(cleaned, collector="A. Smith")
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Literal


# Valid imputation strategies
IMPUTATION_STRATEGIES = ["mean", "median", "drop", "zero", "interpolate"]

# Element columns for cleaning
ELEMENT_COLUMNS = ["Au_ppb", "Cu_ppm", "Pb_ppm", "Zn_ppm", "As_ppm"]


def handle_missing_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    strategy: str = "median",
    threshold: float = 0.5
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Handle missing values in the dataset.

    Supports multiple imputation strategies and tracks the number
    of values imputed per column.

    Args:
        df: Input DataFrame
        columns: Columns to process (default: ELEMENT_COLUMNS)
        strategy: Imputation strategy - one of:
            - "mean": Replace with column mean
            - "median": Replace with column median
            - "drop": Drop rows with missing values
            - "zero": Replace with zero
            - "interpolate": Linear interpolation
        threshold: Maximum fraction of missing values allowed per column
                  (columns exceeding this will be flagged)

    Returns:
        Tuple containing:
            - pd.DataFrame: DataFrame with missing values handled
            - Dict[str, int]: Count of imputed values per column

    Raises:
        ValueError: If strategy is not recognized

    Example:
        >>> cleaned_df, imputed_counts = handle_missing_values(df, strategy="median")
        >>> print(f"Imputed {sum(imputed_counts.values())} total values")
    """
    # TODO: Implement missing value handling
    #
    # Steps:
    # 1. Validate strategy is in IMPUTATION_STRATEGIES
    # 2. Create a copy of the DataFrame
    # 3. If columns is None, use ELEMENT_COLUMNS
    # 4. For each column:
    #    a. Count missing values
    #    b. Check if exceeds threshold (warn if so)
    #    c. Apply imputation based on strategy
    # 5. Return (cleaned_df, imputed_counts_dict)
    #
    # Hints:
    # - Use df[col].isna().sum() to count missing
    # - Use df[col].fillna(df[col].median()) for median imputation
    # - Use df.dropna(subset=columns) for drop strategy

    raise NotImplementedError("TODO: Implement handle_missing_values function")


def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Detect outliers in numerical columns.

    Args:
        df: Input DataFrame
        columns: Columns to check for outliers (default: ELEMENT_COLUMNS)
        method: Detection method - "iqr" (Interquartile Range) or "zscore"
        threshold: For IQR method, multiplier for IQR (default: 1.5)
                  For zscore method, number of standard deviations (default: 3)

    Returns:
        pd.DataFrame: Boolean DataFrame where True indicates an outlier

    Example:
        >>> outliers = detect_outliers(df, method="iqr", threshold=1.5)
        >>> outlier_counts = outliers.sum()
        >>> print(f"Found {outliers.sum().sum()} total outliers")
    """
    # TODO: Implement outlier detection
    #
    # Steps:
    # 1. If columns is None, use ELEMENT_COLUMNS
    # 2. Create empty boolean DataFrame
    # 3. For each column:
    #    - If method == "iqr":
    #      a. Calculate Q1, Q3, IQR
    #      b. Mark values outside (Q1 - threshold*IQR, Q3 + threshold*IQR)
    #    - If method == "zscore":
    #      a. Calculate z-scores
    #      b. Mark values with |z| > threshold
    # 4. Return outlier boolean DataFrame
    #
    # Hints:
    # - Use df[col].quantile(0.25) and quantile(0.75) for Q1, Q3
    # - Use (df[col] - df[col].mean()) / df[col].std() for z-score

    raise NotImplementedError("TODO: Implement detect_outliers function")


def handle_outliers(
    df: pd.DataFrame,
    outliers: pd.DataFrame,
    strategy: str = "clip"
) -> pd.DataFrame:
    """
    Handle detected outliers using the specified strategy.

    Args:
        df: Input DataFrame
        outliers: Boolean DataFrame indicating outlier positions
        strategy: How to handle outliers:
            - "clip": Clip values to the outlier threshold boundaries
            - "remove": Remove rows containing any outliers
            - "nan": Replace outliers with NaN
            - "keep": Keep outliers unchanged (just flag them)

    Returns:
        pd.DataFrame: DataFrame with outliers handled

    Example:
        >>> outliers = detect_outliers(df)
        >>> cleaned = handle_outliers(df, outliers, strategy="clip")
    """
    # TODO: Implement outlier handling
    #
    # Steps:
    # 1. Create a copy of the DataFrame
    # 2. Based on strategy:
    #    - "clip": Clip values using quantile-based bounds
    #    - "remove": Drop rows where any outlier is True
    #    - "nan": Set outlier values to np.nan
    #    - "keep": Return original DataFrame
    # 3. Return modified DataFrame

    raise NotImplementedError("TODO: Implement handle_outliers function")


def filter_by_criteria(
    df: pd.DataFrame,
    collector: Optional[str] = None,
    elevation_range: Optional[Tuple[float, float]] = None,
    sample_types: Optional[List[str]] = None,
    quality_levels: Optional[List[str]] = None,
    date_range: Optional[Tuple[str, str]] = None
) -> pd.DataFrame:
    """
    Filter data based on various criteria.

    All filters are combined with AND logic (all must be satisfied).

    Args:
        df: Input DataFrame
        collector: Filter to samples from this collector
        elevation_range: Tuple of (min, max) elevation in meters
        sample_types: List of sample types to include
        quality_levels: List of quality levels to include
        date_range: Tuple of (start_date, end_date) strings

    Returns:
        pd.DataFrame: Filtered DataFrame

    Example:
        >>> filtered = filter_by_criteria(
        ...     df,
        ...     collector="A. Smith",
        ...     elevation_range=(1200, 1600),
        ...     sample_types=["soil", "rock"]
        ... )
    """
    # TODO: Implement data filtering
    #
    # Steps:
    # 1. Create a copy of the DataFrame
    # 2. Apply each filter if its parameter is not None:
    #    - collector: df['collector'] == collector
    #    - elevation_range: df['elevation'].between(min, max)
    #    - sample_types: df['sample_type'].isin(sample_types)
    #    - quality_levels: df['sample_quality'].isin(quality_levels)
    #    - date_range: df['collection_date'].between(start, end)
    # 3. Return filtered DataFrame
    #
    # Hints:
    # - Chain filters using boolean indexing
    # - Be careful with None checks for each parameter

    raise NotImplementedError("TODO: Implement filter_by_criteria function")


def standardize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize data types for all columns.

    Ensures:
    - Numeric columns are float64
    - Date columns are datetime
    - String columns are stripped of whitespace

    Args:
        df: Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with standardized types
    """
    # TODO: Implement data type standardization
    #
    # Steps:
    # 1. Create a copy of the DataFrame
    # 2. Convert element columns to float64
    # 3. Convert coordinate columns (utm_e, utm_n, elevation) to float64
    # 4. Convert collection_date to datetime
    # 5. Strip whitespace from string columns
    # 6. Return standardized DataFrame

    raise NotImplementedError("TODO: Implement standardize_data_types function")


def clean_data(
    df: pd.DataFrame,
    missing_strategy: str = "median",
    handle_outliers_flag: bool = True,
    outlier_strategy: str = "clip",
    standardize_types: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main cleaning function that orchestrates all cleaning steps.

    Args:
        df: Input DataFrame
        missing_strategy: Strategy for handling missing values
        handle_outliers_flag: Whether to detect and handle outliers
        outlier_strategy: Strategy for handling outliers
        standardize_types: Whether to standardize data types

    Returns:
        Tuple containing:
            - pd.DataFrame: Cleaned DataFrame
            - Dict[str, Any]: Cleaning report with statistics

    Example:
        >>> cleaned_df, report = clean_data(raw_df)
        >>> print(f"Original: {report['original_rows']}, Cleaned: {report['final_rows']}")
    """
    # TODO: Implement main cleaning orchestration
    #
    # Steps:
    # 1. Initialize cleaning report dictionary
    # 2. Store original row count
    # 3. If standardize_types: call standardize_data_types()
    # 4. Call handle_missing_values() with missing_strategy
    # 5. If handle_outliers_flag:
    #    a. Call detect_outliers()
    #    b. Call handle_outliers() with outlier_strategy
    # 6. Store final row count
    # 7. Calculate and store rows removed, missing values imputed, etc.
    # 8. Return (cleaned_df, report)

    raise NotImplementedError("TODO: Implement clean_data function")
