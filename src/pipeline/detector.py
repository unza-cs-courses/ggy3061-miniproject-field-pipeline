"""
Anomaly Detection Module
========================

This module implements anomaly detection for geochemical data,
identifying samples with unusually high element concentrations.

Classes:
    AnomalyDetector: Configurable anomaly detection class

Functions:
    detect_anomalies: Simple function for percentile-based detection
    calculate_thresholds: Calculate anomaly thresholds for elements
    get_anomaly_spatial_context: Add spatial context to detected anomalies

Example:
    >>> from pipeline.detector import detect_anomalies, AnomalyDetector
    >>> anomalies = detect_anomalies(df, percentile=95)
    >>> detector = AnomalyDetector(percentile=90)
    >>> results = detector.fit_detect(df)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field


# Default element columns for anomaly detection
DEFAULT_ELEMENTS = ["Au_ppb", "Cu_ppm", "Pb_ppm", "Zn_ppm", "As_ppm"]


@dataclass
class AnomalyResult:
    """Container for anomaly detection results."""
    sample_id: str
    utm_e: float
    utm_n: float
    elevation: float
    element: str
    value: float
    threshold: float
    percentile_rank: float
    is_anomaly: bool


@dataclass
class AnomalyDetector:
    """
    Configurable anomaly detector for geochemical data.

    Supports multiple detection methods and threshold configurations.

    Attributes:
        percentile: Percentile threshold for anomaly detection (default: 95)
        elements: List of elements to check (default: DEFAULT_ELEMENTS)
        method: Detection method - "percentile" or "zscore" (default: "percentile")
        zscore_threshold: Z-score threshold if method is "zscore" (default: 3.0)

    Example:
        >>> detector = AnomalyDetector(percentile=90, elements=["Au_ppb", "Cu_ppm"])
        >>> detector.fit(df)
        >>> anomalies = detector.detect(df)
    """
    percentile: int = 95
    elements: List[str] = field(default_factory=lambda: DEFAULT_ELEMENTS.copy())
    method: str = "percentile"
    zscore_threshold: float = 3.0

    # Internal state (set during fit)
    _thresholds: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _statistics: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False, repr=False)
    _is_fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """
        Fit the detector to training data.

        Calculates thresholds for each element based on the specified method.

        Args:
            df: Training DataFrame

        Returns:
            self (for method chaining)

        Example:
            >>> detector = AnomalyDetector(percentile=95)
            >>> detector.fit(training_data)
        """
        # TODO: Implement fit method
        #
        # Steps:
        # 1. Filter elements to those present in DataFrame
        # 2. For each element:
        #    a. Calculate statistics (mean, std, percentiles)
        #    b. Calculate threshold based on method:
        #       - "percentile": np.percentile(data, self.percentile)
        #       - "zscore": mean + self.zscore_threshold * std
        #    c. Store threshold in self._thresholds
        #    d. Store statistics in self._statistics
        # 3. Set self._is_fitted = True
        # 4. Return self
        #
        # Hints:
        # - Use df[element].dropna() before calculations
        # - Store statistics as dict: {"mean": ..., "std": ..., "threshold": ...}

        raise NotImplementedError("TODO: Implement fit method")

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in new data using fitted thresholds.

        Args:
            df: DataFrame to detect anomalies in

        Returns:
            pd.DataFrame: DataFrame containing only anomalous samples
                         with columns: sample_id, utm_e, utm_n, elevation,
                         element, value, threshold, percentile_rank

        Raises:
            ValueError: If detector has not been fitted

        Example:
            >>> detector.fit(training_data)
            >>> anomalies = detector.detect(test_data)
            >>> print(f"Found {len(anomalies)} anomalous samples")
        """
        # TODO: Implement detect method
        #
        # Steps:
        # 1. Check self._is_fitted, raise ValueError if False
        # 2. Initialize empty list for anomaly records
        # 3. For each element in self.elements:
        #    a. Get threshold from self._thresholds
        #    b. Find rows where value > threshold
        #    c. For each anomalous row, create record with:
        #       - sample_id, utm_e, utm_n, elevation
        #       - element name, value, threshold
        #       - percentile rank of the value
        # 4. Create DataFrame from anomaly records
        # 5. Return anomaly DataFrame (sorted by value descending)
        #
        # Hints:
        # - Use scipy.stats.percentileofscore() for percentile rank
        # - Or calculate manually: (value > all_values).mean() * 100

        raise NotImplementedError("TODO: Implement detect method")

    def fit_detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit detector and detect anomalies in one step.

        Args:
            df: DataFrame to fit and detect on

        Returns:
            pd.DataFrame: Anomaly DataFrame

        Example:
            >>> detector = AnomalyDetector(percentile=95)
            >>> anomalies = detector.fit_detect(df)
        """
        return self.fit(df).detect(df)

    def get_thresholds(self) -> Dict[str, float]:
        """
        Get calculated thresholds for each element.

        Returns:
            Dict mapping element names to threshold values

        Raises:
            ValueError: If detector has not been fitted
        """
        if not self._is_fitted:
            raise ValueError("Detector has not been fitted. Call fit() first.")
        return self._thresholds.copy()

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get calculated statistics for each element.

        Returns:
            Dict mapping element names to statistics dictionaries

        Raises:
            ValueError: If detector has not been fitted
        """
        if not self._is_fitted:
            raise ValueError("Detector has not been fitted. Call fit() first.")
        return {k: v.copy() for k, v in self._statistics.items()}


def detect_anomalies(
    df: pd.DataFrame,
    percentile: int = 95,
    elements: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Simple function for percentile-based anomaly detection.

    This is a convenience function that wraps AnomalyDetector.

    Args:
        df: Input DataFrame
        percentile: Percentile threshold (default: 95)
        elements: Elements to check (default: DEFAULT_ELEMENTS)

    Returns:
        pd.DataFrame: Anomaly DataFrame with sample details

    Example:
        >>> anomalies = detect_anomalies(df, percentile=90)
        >>> high_gold = anomalies[anomalies["element"] == "Au_ppb"]
    """
    # TODO: Implement simple anomaly detection function
    #
    # Steps:
    # 1. If elements is None, use DEFAULT_ELEMENTS
    # 2. Create AnomalyDetector with given percentile and elements
    # 3. Call fit_detect() and return result

    raise NotImplementedError("TODO: Implement detect_anomalies function")


def calculate_thresholds(
    df: pd.DataFrame,
    percentile: int = 95,
    elements: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate anomaly thresholds for each element.

    Args:
        df: Input DataFrame
        percentile: Percentile to use as threshold
        elements: Elements to calculate thresholds for

    Returns:
        Dict mapping element names to threshold values

    Example:
        >>> thresholds = calculate_thresholds(df, percentile=95)
        >>> print(f"Au threshold: {thresholds['Au_ppb']:.2f} ppb")
    """
    # TODO: Implement threshold calculation
    #
    # Steps:
    # 1. If elements is None, use DEFAULT_ELEMENTS
    # 2. For each element present in DataFrame:
    #    a. Calculate the specified percentile
    #    b. Store in results dictionary
    # 3. Return thresholds dictionary
    #
    # Hints:
    # - Use df[element].quantile(percentile / 100)
    # - Or np.percentile(df[element].dropna(), percentile)

    raise NotImplementedError("TODO: Implement calculate_thresholds function")


def get_anomaly_spatial_context(
    anomalies: pd.DataFrame,
    df: pd.DataFrame,
    radius: float = 500.0
) -> pd.DataFrame:
    """
    Add spatial context to detected anomalies.

    For each anomaly, finds nearby samples and calculates:
    - Number of nearby samples
    - Average values of nearby samples
    - Whether there are clustered anomalies

    Args:
        anomalies: DataFrame of detected anomalies
        df: Full sample DataFrame
        radius: Search radius in meters (default: 500)

    Returns:
        pd.DataFrame: Anomaly DataFrame with added spatial context columns

    Example:
        >>> anomalies = detect_anomalies(df)
        >>> with_context = get_anomaly_spatial_context(anomalies, df)
        >>> clustered = with_context[with_context["nearby_anomalies"] > 2]
    """
    # TODO: Implement spatial context calculation
    #
    # Steps:
    # 1. Create copy of anomalies DataFrame
    # 2. For each anomaly:
    #    a. Calculate distance to all samples using UTM coordinates
    #    b. Find samples within radius
    #    c. Count nearby samples
    #    d. Calculate average element value of nearby samples
    #    e. Count how many nearby samples are also anomalies
    # 3. Add new columns to DataFrame
    # 4. Return enhanced DataFrame
    #
    # Hints:
    # - Distance = sqrt((e1-e2)^2 + (n1-n2)^2)
    # - Use numpy broadcasting for efficient distance calculation

    raise NotImplementedError("TODO: Implement get_anomaly_spatial_context function")


def summarize_anomalies(
    anomalies: pd.DataFrame,
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Generate a summary of detected anomalies.

    Args:
        anomalies: DataFrame of detected anomalies
        df: Full sample DataFrame (for calculating percentages)

    Returns:
        Dict containing:
            - total_anomalies: Total number of anomalous samples
            - by_element: Count of anomalies per element
            - spatial_extent: Bounding box of anomaly locations
            - percentage: Percent of total samples that are anomalies

    Example:
        >>> summary = summarize_anomalies(anomalies, df)
        >>> print(f"Found {summary['total_anomalies']} anomalies ({summary['percentage']:.1f}%)")
    """
    # TODO: Implement anomaly summary
    #
    # Steps:
    # 1. Count total unique anomalous samples
    # 2. Count anomalies per element
    # 3. Calculate bounding box (min/max UTM coordinates)
    # 4. Calculate percentage of total samples
    # 5. Return summary dictionary

    raise NotImplementedError("TODO: Implement summarize_anomalies function")
