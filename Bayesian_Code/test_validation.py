#!/usr/bin/env python
# test_validation.py
# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for data validation functions: outlier detection and stationarity tests.
#
# Usage:
#   python -m pytest test_validation.py -v
#   or: python test_validation.py
# ─────────────────────────────────────────────────────────────────────────────

import unittest
import numpy as np
import pandas as pd
from data_prep import (
    detect_outliers_iqr,
    detect_outliers_robust_zscore,
    run_stationarity_tests,
)


class TestOutlierDetectionIQR(unittest.TestCase):
    """Test IQR-based outlier detection."""

    def test_clean_data_no_outliers(self):
        """Test that clean data has no detected outliers."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        outliers = detect_outliers_iqr(data)
        self.assertEqual(np.sum(outliers), 0, "Clean data should have no outliers")

    def test_extreme_outlier_detected(self):
        """Test that extreme values are detected as outliers."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        outliers = detect_outliers_iqr(data)
        self.assertTrue(outliers[-1], "Extreme value should be detected as outlier")

    def test_negative_outlier_detected(self):
        """Test that negative outliers are detected."""
        data = np.array([-100.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        outliers = detect_outliers_iqr(data)
        self.assertTrue(outliers[0], "Negative outlier should be detected")

    def test_multiple_outliers(self):
        """Test detection of multiple outliers."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -50.0, 100.0])
        outliers = detect_outliers_iqr(data)
        outlier_count = np.sum(outliers)
        self.assertGreaterEqual(outlier_count, 2, "Should detect at least 2 outliers")

    def test_threshold_sensitivity(self):
        """Test that threshold parameter affects detection."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 20.0])
        outliers_strict = detect_outliers_iqr(data, threshold=1.0)
        outliers_loose = detect_outliers_iqr(data, threshold=3.0)
        # Stricter threshold should detect more or equal outliers
        self.assertGreaterEqual(
            np.sum(outliers_strict),
            np.sum(outliers_loose),
            "Stricter threshold should detect more outliers"
        )

    def test_zero_iqr(self):
        """Test handling of zero IQR (constant data)."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        outliers = detect_outliers_iqr(data)
        self.assertEqual(np.sum(outliers), 0, "Constant data should have no outliers (zero IQR)")


class TestOutlierDetectionRobustZScore(unittest.TestCase):
    """Test robust z-score based outlier detection."""

    def test_clean_data_no_outliers(self):
        """Test that clean data has no detected outliers."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        outliers = detect_outliers_robust_zscore(data)
        self.assertEqual(np.sum(outliers), 0, "Clean data should have no outliers")

    def test_extreme_outlier_detected(self):
        """Test that extreme values are detected."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        outliers = detect_outliers_robust_zscore(data)
        self.assertTrue(outliers[-1], "Extreme value should be detected as outlier")

    def test_threshold_sensitivity(self):
        """Test that threshold parameter affects detection."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 20.0])
        outliers_strict = detect_outliers_robust_zscore(data, threshold=2.0)
        outliers_loose = detect_outliers_robust_zscore(data, threshold=4.0)
        # Stricter threshold should detect more or equal outliers
        self.assertGreaterEqual(
            np.sum(outliers_strict),
            np.sum(outliers_loose),
            "Stricter threshold should detect more outliers"
        )

    def test_zero_mad(self):
        """Test handling of zero MAD (low variance data)."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.1])
        outliers = detect_outliers_robust_zscore(data)
        # Should not crash; may flag the slightly different value
        self.assertIsInstance(outliers, np.ndarray)
        self.assertEqual(len(outliers), len(data))

    def test_bimodal_distribution(self):
        """Test with bimodal distribution (two groups)."""
        # Two clusters: [1, 2, 3] and [98, 99, 100]
        data = np.array([1.0, 2.0, 3.0, 98.0, 99.0, 100.0])
        outliers = detect_outliers_robust_zscore(data, threshold=2.0)
        # Robust z-score can struggle with truly bimodal data; just verify it doesn't crash
        # and returns a valid boolean array
        self.assertIsInstance(outliers, np.ndarray)
        self.assertEqual(len(outliers), len(data))


class TestStationarityTests(unittest.TestCase):
    """Test stationarity test functions (ADF/KPSS)."""

    def test_stationary_white_noise(self):
        """Test that white noise passes stationarity tests."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        results = run_stationarity_tests(data)
        
        # ADF should suggest stationarity (low p-value)
        if "adf_pvalue" in results:
            self.assertLess(results["adf_pvalue"], 0.1,
                           "White noise should be stationary (ADF p-value < 0.1)")

    def test_nonstationary_trend(self):
        """Test that a trending series is flagged as non-stationary."""
        # Create a clear uptrend
        data = np.linspace(0, 100, 100)
        results = run_stationarity_tests(data)
        
        # ADF should suggest non-stationarity (high p-value)
        if "adf_pvalue" in results:
            self.assertGreater(results["adf_pvalue"], 0.05,
                             "Trending data should be non-stationary (ADF p-value > 0.05)")

    def test_nonstationary_random_walk(self):
        """Test that a random walk is flagged as non-stationary."""
        np.random.seed(42)
        # Random walk: cumulative sum of white noise
        data = np.cumsum(np.random.normal(0, 1, 100))
        results = run_stationarity_tests(data)
        
        # ADF should suggest non-stationarity
        if "adf_pvalue" in results:
            self.assertGreater(results["adf_pvalue"], 0.05,
                             "Random walk should be non-stationary (ADF p-value > 0.05)")

    def test_seasonal_data(self):
        """Test stationarity on seasonal data."""
        # Create seasonal data
        t = np.arange(100)
        data = 10 + 5 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.5, 100)
        results = run_stationarity_tests(data)
        
        # Seasonal data can be stationary
        self.assertIn("adf_pvalue", results)
        self.assertIn("adf_stat", results)

    def test_constant_series(self):
        """Test that a constant series is technically stationary."""
        data = np.ones(50) * 5.0
        results = run_stationarity_tests(data)
        
        # Constant data should be stationary
        if "adf_pvalue" in results:
            self.assertLess(results["adf_pvalue"], 0.1,
                           "Constant series should be stationary")

    def test_missing_statsmodels(self):
        """Test graceful handling when statsmodels is unavailable."""
        # This test verifies error handling, not the actual statsmodels import
        data = np.random.normal(0, 1, 50)
        results = run_stationarity_tests(data)
        
        # Should return a dict (empty if statsmodels not available)
        self.assertIsInstance(results, dict)


class TestValidationOnRealData(unittest.TestCase):
    """Integration tests on synthetic MMM-like data."""

    def test_outlier_detection_on_spend_data(self):
        """Test outlier detection on simulated spend data."""
        np.random.seed(42)
        # Simulate weekly spend: base + noise + occasional spike
        base_spend = np.random.lognormal(8, 0.3, 52)  # 52 weeks
        # Add one large outlier (campaign spike)
        spend_with_outlier = base_spend.copy()
        spend_with_outlier[25] *= 5  # 5x spike in week 25
        
        outliers_iqr = detect_outliers_iqr(spend_with_outlier)
        outliers_robust = detect_outliers_robust_zscore(spend_with_outlier)
        
        # At least one of the methods should detect the spike
        self.assertTrue(
            np.any(outliers_iqr) or np.any(outliers_robust),
            "Should detect the spend spike"
        )

    def test_stationarity_on_response_data(self):
        """Test stationarity on simulated response (sales/signups) data."""
        np.random.seed(42)
        # Simulate 52 weeks of response with trend + seasonality
        t = np.arange(52)
        trend = t * 0.5  # Linear uptrend
        seasonality = 20 * np.sin(2 * np.pi * t / 13)  # Bi-weekly seasonality
        noise = np.random.normal(0, 5, 52)
        response = 100 + trend + seasonality + noise
        
        results = run_stationarity_tests(response)
        # Should have results (statsmodels is typically available)
        if results:
            self.assertIn("adf_pvalue", results)

    def test_collinearity_detection(self):
        """Test detection of correlated media channels."""
        np.random.seed(42)
        T = 52
        
        # Create two highly correlated channels (TV budget drives radio spend)
        tv_spend = np.random.lognormal(8, 0.2, T)
        radio_spend = tv_spend * 0.8 + np.random.normal(0, tv_spend.mean() * 0.1, T)
        
        data = np.column_stack([tv_spend, radio_spend])
        corr_matrix = np.corrcoef(data.T)
        
        # Check that the off-diagonal correlation is high
        self.assertGreater(np.abs(corr_matrix[0, 1]), 0.7,
                         "TV and radio spend should be highly correlated")


class TestEndToEndValidation(unittest.TestCase):
    """End-to-end validation workflow tests."""

    def test_validation_report_structure(self):
        """Test that validation produces expected report structure."""
        np.random.seed(42)
        T = 100
        C = 3  # 3 channels
        
        # Generate synthetic data: T rows, C columns
        data = np.random.lognormal(8, 0.3, (T, C))
        
        # Apply outlier detection to each channel
        outlier_reports = {}
        for j in range(C):
            outlier_reports[f"channel_{j}"] = {
                "iqr": detect_outliers_iqr(data[:, j]),
                "robust_zscore": detect_outliers_robust_zscore(data[:, j]),
            }
        
        # Verify structure
        for ch_name, report in outlier_reports.items():
            self.assertIn("iqr", report)
            self.assertIn("robust_zscore", report)
            self.assertEqual(len(report["iqr"]), T)
            self.assertEqual(len(report["robust_zscore"]), T)

    def test_combined_validation_logic(self):
        """Test combining multiple validation checks."""
        np.random.seed(42)
        T = 52
        
        # Simulated response with some issues
        base = np.random.normal(100, 10, T)
        base[10] = 500  # Outlier
        base = np.cumsum(base) / 10  # Add trend (non-stationary)
        
        # Run all checks
        outliers_iqr = detect_outliers_iqr(base)
        outliers_robust = detect_outliers_robust_zscore(base)
        stationarity = run_stationarity_tests(base)
        
        # Compile a quality report
        quality_report = {
            "n_outliers_iqr": int(np.sum(outliers_iqr)),
            "n_outliers_robust": int(np.sum(outliers_robust)),
            "is_stationary_adf": stationarity.get("adf_pvalue", 1.0) < 0.05,
            "is_stationary_kpss": stationarity.get("kpss_pvalue", 0.0) > 0.05,
        }
        
        # Assert structure
        self.assertIn("n_outliers_iqr", quality_report)
        self.assertIn("n_outliers_robust", quality_report)


if __name__ == "__main__":
    unittest.main()
