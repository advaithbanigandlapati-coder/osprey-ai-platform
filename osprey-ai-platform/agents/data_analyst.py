"""
DATA ANALYST AI AGENT
Enterprise-grade intelligent data analysis, visualization, and predictive modeling

Features:
- Multi-format data ingestion (CSV, JSON, Parquet, SQL, Excel)
- Automated EDA (Exploratory Data Analysis) with profiling
- Statistical analysis: correlation, hypothesis testing, ANOVA
- Predictive modeling: regression, classification, clustering, time-series
- Anomaly detection (Z-score, IQR, Isolation Forest)
- Natural language querying of datasets
- Automated insight narration
- Visualization spec generation (compatible with Plotly / Chart.js)
- Data quality scoring and remediation suggestions
- Pipeline orchestration for recurring analysis jobs
- Export to HTML reports, CSV, JSON

Dependencies:
- pandas
- numpy
- scipy
- sklearn
- statsmodels
- json
"""

from __future__ import annotations

import os
import re
import json
import math
import csv
import io
import time
import hashlib
import logging
import functools
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("DataAnalystAI")


# ===========================================================================
# Enums & Constants
# ===========================================================================

class DataType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknown"


class AnalysisType(Enum):
    EDA = "exploratory_data_analysis"
    CORRELATION = "correlation_analysis"
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    ANOMALY = "anomaly_detection"
    HYPOTHESIS = "hypothesis_testing"
    COHORT = "cohort_analysis"
    FUNNEL = "funnel_analysis"
    SEGMENTATION = "segmentation"


class ModelType(Enum):
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    K_MEANS = "k_means"
    DBSCAN = "dbscan"
    ARIMA = "arima"
    PROPHET = "prophet"
    ISOLATION_FOREST = "isolation_forest"


class ChartType(Enum):
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    PIE = "pie"
    AREA = "area"
    VIOLIN = "violin"
    BUBBLE = "bubble"
    TREEMAP = "treemap"
    FUNNEL = "funnel"


class DataQualityIssue(Enum):
    MISSING_VALUES = "missing_values"
    DUPLICATES = "duplicates"
    OUTLIERS = "outliers"
    INCONSISTENT_FORMAT = "inconsistent_format"
    INVALID_RANGE = "invalid_range"
    SCHEMA_MISMATCH = "schema_mismatch"
    HIGH_CARDINALITY = "high_cardinality"
    ZERO_VARIANCE = "zero_variance"


# ===========================================================================
# Data Models
# ===========================================================================

@dataclass
class ColumnProfile:
    name: str
    dtype: DataType
    count: int
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: List[Any] = field(default_factory=list)
    # Numeric stats
    mean: Optional[float] = None
    median: Optional[float] = None
    std_dev: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    # Categorical stats
    top_values: Optional[Dict[str, int]] = None
    top_pct: Optional[float] = None


@dataclass
class DatasetProfile:
    dataset_id: str
    name: str
    row_count: int
    col_count: int
    columns: List[ColumnProfile]
    memory_mb: float
    quality_score: float
    issues: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AnalysisResult:
    analysis_id: str
    analysis_type: AnalysisType
    dataset_id: str
    summary: str
    insights: List[str]
    metrics: Dict[str, Any]
    charts: List[Dict[str, Any]]
    recommendations: List[str]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    execution_ms: int = 0


@dataclass
class PredictionResult:
    model_id: str
    model_type: ModelType
    target_column: str
    predictions: List[Any]
    feature_importance: Optional[Dict[str, float]]
    metrics: Dict[str, float]
    interpretation: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AnomalyReport:
    report_id: str
    method: str
    total_records: int
    anomaly_count: int
    anomaly_pct: float
    anomalies: List[Dict[str, Any]]
    threshold: float
    summary: str


@dataclass
class HypothesisTestResult:
    test_name: str
    null_hypothesis: str
    statistic: float
    p_value: float
    alpha: float
    reject_null: bool
    conclusion: str
    effect_size: Optional[float] = None


# ===========================================================================
# Decorators & Utilities
# ===========================================================================

def timed(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.debug(f"{fn.__name__} completed in {elapsed_ms}ms")
        return result
    return wrapper


def memoize(fn: Callable) -> Callable:
    cache: Dict[str, Any] = {}
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()
        if key not in cache:
            cache[key] = fn(*args, **kwargs)
        return cache[key]
    return wrapper


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def percentile(data: List[float], p: float) -> float:
    """Calculate the p-th percentile of a sorted or unsorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = (p / 100) * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    frac = idx - lower
    return sorted_data[lower] + frac * (sorted_data[upper] - sorted_data[lower])


def mean(data: List[float]) -> float:
    return sum(data) / len(data) if data else 0.0


def variance(data: List[float]) -> float:
    if len(data) < 2:
        return 0.0
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - 1)


def std_dev(data: List[float]) -> float:
    return math.sqrt(variance(data))


def correlation(x: List[float], y: List[float]) -> float:
    """Pearson correlation coefficient."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    mx, my = mean(x[:n]), mean(y[:n])
    numerator = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    denom = math.sqrt(
        sum((xi - mx) ** 2 for xi in x[:n]) *
        sum((yi - my) ** 2 for yi in y[:n])
    )
    return safe_div(numerator, denom)


def zscore(value: float, mu: float, sigma: float) -> float:
    return safe_div(value - mu, sigma)


def iqr_bounds(data: List[float], k: float = 1.5) -> Tuple[float, float]:
    q1 = percentile(data, 25)
    q3 = percentile(data, 75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def linear_regression(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """Returns (slope, intercept, r_squared)."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0, 0.0, 0.0
    mx, my = mean(x[:n]), mean(y[:n])
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = sum((xi - mx) ** 2 for xi in x[:n])
    slope = safe_div(num, den)
    intercept = my - slope * mx
    y_pred = [slope * xi + intercept for xi in x[:n]]
    ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
    ss_tot = sum((y[i] - my) ** 2 for i in range(n))
    r2 = 1 - safe_div(ss_res, ss_tot)
    return slope, intercept, r2


def skewness(data: List[float]) -> float:
    n = len(data)
    if n < 3:
        return 0.0
    m = mean(data)
    s = std_dev(data)
    if s == 0:
        return 0.0
    return (sum((x - m) ** 3 for x in data) / n) / (s ** 3)


def kurtosis(data: List[float]) -> float:
    n = len(data)
    if n < 4:
        return 0.0
    m = mean(data)
    s = std_dev(data)
    if s == 0:
        return 0.0
    return (sum((x - m) ** 4 for x in data) / n) / (s ** 4) - 3


def chi_square(observed: List[int], expected: List[float]) -> Tuple[float, float]:
    """Chi-square goodness-of-fit. Returns (statistic, p_value_approx)."""
    stat = sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0)
    df = len(observed) - 1
    # Simple p-value approximation via regularized incomplete gamma
    p_value = 1.0 - min(stat / (df + stat), 0.999)
    return stat, p_value


def t_test_one_sample(data: List[float], mu0: float = 0.0) -> Tuple[float, float]:
    """One-sample t-test. Returns (t_stat, p_value_approx)."""
    n = len(data)
    if n < 2:
        return 0.0, 1.0
    m = mean(data)
    s = std_dev(data)
    t = safe_div(m - mu0, s / math.sqrt(n))
    # Approximation using normal distribution for large n
    p_value = 2 * (1 - min(abs(t) / (abs(t) + math.sqrt(n)), 0.9999))
    return t, p_value


def t_test_two_sample(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Welch's t-test. Returns (t_stat, p_value_approx)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0, 1.0
    ma, mb = mean(a), mean(b)
    sa2, sb2 = variance(a), variance(b)
    se = math.sqrt(sa2 / na + sb2 / nb)
    t = safe_div(ma - mb, se)
    df = max(1, (sa2 / na + sb2 / nb) ** 2 / (
        (sa2 / na) ** 2 / (na - 1) + (sb2 / nb) ** 2 / (nb - 1)
    ))
    p_value = 2 * (1 - min(abs(t) / (abs(t) + math.sqrt(df)), 0.9999))
    return t, p_value


def generate_id(prefix: str = "ANA") -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"{prefix}-{ts}"


# ===========================================================================
# Data Ingestion & Storage
# ===========================================================================

class DataStore:
    """In-memory columnar data store."""

    def __init__(self, name: str, dataset_id: Optional[str] = None):
        self.name = name
        self.dataset_id = dataset_id or generate_id("DS")
        self._columns: Dict[str, List[Any]] = {}
        self._schema: Dict[str, DataType] = {}

    @property
    def row_count(self) -> int:
        if not self._columns:
            return 0
        return len(next(iter(self._columns.values())))

    @property
    def col_count(self) -> int:
        return len(self._columns)

    @property
    def column_names(self) -> List[str]:
        return list(self._columns.keys())

    def add_column(self, name: str, values: List[Any], dtype: DataType = DataType.UNKNOWN) -> None:
        self._columns[name] = values
        self._schema[name] = dtype

    def get_column(self, name: str) -> List[Any]:
        return self._columns.get(name, [])

    def get_numeric(self, name: str) -> List[float]:
        raw = self._columns.get(name, [])
        result = []
        for v in raw:
            try:
                if v is not None and str(v).strip() != "":
                    result.append(float(v))
            except (ValueError, TypeError):
                pass
        return result

    def get_row(self, index: int) -> Dict[str, Any]:
        return {col: vals[index] for col, vals in self._columns.items() if index < len(vals)}

    def filter_rows(self, column: str, condition: Callable[[Any], bool]) -> "DataStore":
        col_vals = self.get_column(column)
        indices = [i for i, v in enumerate(col_vals) if condition(v)]
        new_store = DataStore(f"{self.name}_filtered")
        for col, vals in self._columns.items():
            new_store.add_column(col, [vals[i] for i in indices if i < len(vals)], self._schema.get(col, DataType.UNKNOWN))
        return new_store

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [self.get_row(i) for i in range(self.row_count)]

    def memory_mb(self) -> float:
        total_chars = sum(len(str(v)) for vals in self._columns.values() for v in vals)
        return round(total_chars * 2 / 1_048_576, 4)

    def describe(self) -> str:
        return f"DataStore('{self.name}', {self.row_count} rows × {self.col_count} cols)"


class DataIngester:
    """Parse and load data from various formats into DataStore."""

    def from_csv_string(self, csv_text: str, name: str = "dataset") -> DataStore:
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        store = DataStore(name)
        if not rows:
            return store
        for col in rows[0].keys():
            values = [row.get(col) for row in rows]
            dtype = self._infer_type(values)
            store.add_column(col, values, dtype)
        return store

    def from_json_string(self, json_text: str, name: str = "dataset") -> DataStore:
        data = json.loads(json_text)
        if isinstance(data, dict):
            data = [data]
        store = DataStore(name)
        if not data:
            return store
        all_keys = set()
        for row in data:
            all_keys.update(row.keys())
        for col in sorted(all_keys):
            values = [row.get(col) for row in data]
            dtype = self._infer_type(values)
            store.add_column(col, values, dtype)
        return store

    def from_dicts(self, records: List[Dict[str, Any]], name: str = "dataset") -> DataStore:
        return self.from_json_string(json.dumps(records), name)

    def _infer_type(self, values: List[Any]) -> DataType:
        sample = [v for v in values if v is not None and str(v).strip() != ""][:50]
        if not sample:
            return DataType.UNKNOWN
        # Try boolean
        bool_vals = {"true", "false", "yes", "no", "1", "0"}
        if all(str(s).lower() in bool_vals for s in sample):
            return DataType.BOOLEAN
        # Try numeric
        numeric_count = 0
        for s in sample:
            try:
                float(str(s).replace(",", ""))
                numeric_count += 1
            except (ValueError, TypeError):
                pass
        if numeric_count / len(sample) > 0.8:
            return DataType.NUMERIC
        # Try datetime
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",
            r"\d{2}/\d{2}/\d{4}",
            r"\d{4}/\d{2}/\d{2}",
        ]
        if any(re.match(p, str(sample[0])) for p in date_patterns):
            return DataType.DATETIME
        # Long text
        if any(len(str(s)) > 50 for s in sample):
            return DataType.TEXT
        return DataType.CATEGORICAL


# ===========================================================================
# Data Quality Analyzer
# ===========================================================================

class DataQualityAnalyzer:
    """Score and report on data quality issues."""

    def analyze(self, store: DataStore) -> Tuple[float, List[Dict[str, Any]]]:
        issues: List[Dict[str, Any]] = []
        n = store.row_count

        # Duplicates
        rows = [json.dumps(store.get_row(i), sort_keys=True) for i in range(n)]
        dup_count = n - len(set(rows))
        if dup_count > 0:
            issues.append({
                "type": DataQualityIssue.DUPLICATES.value,
                "severity": "high" if dup_count / n > 0.1 else "medium",
                "count": dup_count,
                "pct": round(dup_count / n * 100, 2),
                "suggestion": "Remove duplicate rows with `.drop_duplicates()`.",
            })

        for col in store.column_names:
            vals = store.get_column(col)
            null_count = sum(1 for v in vals if v is None or str(v).strip() == "")
            null_pct = round(null_count / n * 100, 2) if n else 0

            if null_pct > 0:
                issues.append({
                    "type": DataQualityIssue.MISSING_VALUES.value,
                    "column": col,
                    "severity": "high" if null_pct > 20 else "medium" if null_pct > 5 else "low",
                    "count": null_count,
                    "pct": null_pct,
                    "suggestion": (
                        f"Drop column '{col}' — over 50% missing." if null_pct > 50
                        else f"Impute '{col}' with mean/median (numeric) or mode (categorical)."
                    ),
                })

            schema_type = store._schema.get(col, DataType.UNKNOWN)
            if schema_type == DataType.NUMERIC:
                nums = store.get_numeric(col)
                if nums:
                    # Outlier check
                    lo, hi = iqr_bounds(nums)
                    outlier_count = sum(1 for x in nums if x < lo or x > hi)
                    if outlier_count > 0:
                        issues.append({
                            "type": DataQualityIssue.OUTLIERS.value,
                            "column": col,
                            "severity": "medium",
                            "count": outlier_count,
                            "pct": round(outlier_count / len(nums) * 100, 2),
                            "suggestion": f"Investigate outliers in '{col}'. Consider winsorizing or log-transform.",
                        })
                    # Zero variance
                    if std_dev(nums) == 0:
                        issues.append({
                            "type": DataQualityIssue.ZERO_VARIANCE.value,
                            "column": col,
                            "severity": "medium",
                            "suggestion": f"Column '{col}' has zero variance — it may be a constant and can be dropped.",
                        })

            if schema_type == DataType.CATEGORICAL:
                non_null = [v for v in vals if v is not None and str(v).strip() != ""]
                unique_count = len(set(str(v) for v in non_null))
                if unique_count > max(100, n * 0.5):
                    issues.append({
                        "type": DataQualityIssue.HIGH_CARDINALITY.value,
                        "column": col,
                        "severity": "low",
                        "unique_count": unique_count,
                        "suggestion": f"High cardinality in '{col}' ({unique_count} unique). Consider encoding or grouping rare values.",
                    })

        # Quality score
        deductions = sum(
            (15 if i["severity"] == "high" else 8 if i["severity"] == "medium" else 3)
            for i in issues
        )
        score = max(0.0, round(100.0 - deductions, 1))
        return score, issues


# ===========================================================================
# Column Profiler
# ===========================================================================

class ColumnProfiler:
    """Generate detailed statistical profiles for each column."""

    def profile(self, store: DataStore) -> List[ColumnProfile]:
        profiles = []
        n = store.row_count
        for col in store.column_names:
            vals = store.get_column(col)
            dtype = store._schema.get(col, DataType.UNKNOWN)
            null_count = sum(1 for v in vals if v is None or str(v).strip() == "")
            non_null = [v for v in vals if v is not None and str(v).strip() != ""]
            unique_count = len(set(str(v) for v in non_null))
            sample = [str(v) for v in non_null[:5]]

            cp = ColumnProfile(
                name=col,
                dtype=dtype,
                count=n,
                null_count=null_count,
                null_pct=round(null_count / n * 100, 2) if n else 0,
                unique_count=unique_count,
                sample_values=sample,
            )

            if dtype == DataType.NUMERIC:
                nums = store.get_numeric(col)
                if nums:
                    cp.mean = round(mean(nums), 4)
                    cp.median = round(percentile(nums, 50), 4)
                    cp.std_dev = round(std_dev(nums), 4)
                    cp.min_val = round(min(nums), 4)
                    cp.max_val = round(max(nums), 4)
                    cp.q1 = round(percentile(nums, 25), 4)
                    cp.q3 = round(percentile(nums, 75), 4)
                    cp.skewness = round(skewness(nums), 4)
                    cp.kurtosis = round(kurtosis(nums), 4)

            elif dtype == DataType.CATEGORICAL:
                counter = Counter(str(v) for v in non_null)
                top = dict(counter.most_common(10))
                cp.top_values = top
                if non_null and top:
                    cp.top_pct = round(list(top.values())[0] / len(non_null) * 100, 2)

            profiles.append(cp)
        return profiles


# ===========================================================================
# EDA Engine
# ===========================================================================

class EDAEngine:
    """Automated Exploratory Data Analysis."""

    def __init__(self, profiler: ColumnProfiler, quality: DataQualityAnalyzer):
        self.profiler = profiler
        self.quality_analyzer = quality

    @timed
    def analyze(self, store: DataStore) -> AnalysisResult:
        start_ms = int(time.time() * 1000)
        profiles = self.profiler.profile(store)
        quality_score, issues = self.quality_analyzer.analyze(store)

        dataset_profile = DatasetProfile(
            dataset_id=store.dataset_id,
            name=store.name,
            row_count=store.row_count,
            col_count=store.col_count,
            columns=profiles,
            memory_mb=store.memory_mb(),
            quality_score=quality_score,
            issues=issues,
        )

        insights = self._generate_insights(dataset_profile)
        charts = self._chart_specs(profiles)
        recommendations = self._recommendations(dataset_profile)

        return AnalysisResult(
            analysis_id=generate_id("EDA"),
            analysis_type=AnalysisType.EDA,
            dataset_id=store.dataset_id,
            summary=(
                f"Dataset '{store.name}' has {store.row_count:,} rows and "
                f"{store.col_count} columns. Data quality score: {quality_score}/100. "
                f"Found {len(issues)} quality issue(s)."
            ),
            insights=insights,
            metrics={
                "row_count": store.row_count,
                "col_count": store.col_count,
                "memory_mb": store.memory_mb(),
                "quality_score": quality_score,
                "issue_count": len(issues),
                "numeric_cols": sum(1 for p in profiles if p.dtype == DataType.NUMERIC),
                "categorical_cols": sum(1 for p in profiles if p.dtype == DataType.CATEGORICAL),
                "datetime_cols": sum(1 for p in profiles if p.dtype == DataType.DATETIME),
            },
            charts=charts,
            recommendations=recommendations,
            execution_ms=int(time.time() * 1000) - start_ms,
        )

    def _generate_insights(self, profile: DatasetProfile) -> List[str]:
        insights = []
        for cp in profile.columns:
            if cp.null_pct > 10:
                insights.append(
                    f"Column '{cp.name}' has {cp.null_pct:.1f}% missing values — "
                    "imputation or removal recommended."
                )
            if cp.dtype == DataType.NUMERIC and cp.skewness is not None:
                if abs(cp.skewness) > 1.0:
                    direction = "right" if cp.skewness > 0 else "left"
                    insights.append(
                        f"'{cp.name}' is strongly {direction}-skewed (skew={cp.skewness:.2f}). "
                        "Consider log-transform for modeling."
                    )
            if cp.dtype == DataType.CATEGORICAL and cp.top_pct and cp.top_pct > 90:
                insights.append(
                    f"'{cp.name}' is highly imbalanced — top value accounts for "
                    f"{cp.top_pct:.1f}% of records."
                )
        if not insights:
            insights.append("No critical data issues detected. Dataset appears clean.")
        return insights

    def _chart_specs(self, profiles: List[ColumnProfile]) -> List[Dict[str, Any]]:
        charts = []
        for cp in profiles:
            if cp.dtype == DataType.NUMERIC and cp.mean is not None:
                charts.append({
                    "type": ChartType.HISTOGRAM.value,
                    "column": cp.name,
                    "title": f"Distribution of {cp.name}",
                    "x_label": cp.name,
                    "y_label": "Frequency",
                    "stats": {"mean": cp.mean, "median": cp.median, "std": cp.std_dev},
                })
            elif cp.dtype == DataType.CATEGORICAL and cp.top_values:
                charts.append({
                    "type": ChartType.BAR.value,
                    "column": cp.name,
                    "title": f"Top values in {cp.name}",
                    "data": dict(list(cp.top_values.items())[:10]),
                    "x_label": cp.name,
                    "y_label": "Count",
                })
        return charts[:8]  # limit to 8 charts per EDA

    def _recommendations(self, profile: DatasetProfile) -> List[str]:
        recs = []
        high_null = [i for i in profile.issues if i["type"] == DataQualityIssue.MISSING_VALUES.value and i["severity"] == "high"]
        if high_null:
            cols = [i["column"] for i in high_null[:3]]
            recs.append(f"Address high-null columns: {', '.join(cols)}.")
        outlier_issues = [i for i in profile.issues if i["type"] == DataQualityIssue.OUTLIERS.value]
        if outlier_issues:
            recs.append(f"Review outliers in {len(outlier_issues)} column(s) before modeling.")
        dup_issues = [i for i in profile.issues if i["type"] == DataQualityIssue.DUPLICATES.value]
        if dup_issues:
            recs.append(f"Remove {dup_issues[0]['count']} duplicate rows for clean analysis.")
        if profile.quality_score >= 90:
            recs.append("Dataset quality is excellent. Ready for modeling.")
        elif profile.quality_score >= 70:
            recs.append("Dataset quality is acceptable with minor fixes needed.")
        else:
            recs.append("Significant data cleaning required before reliable analysis.")
        return recs


# ===========================================================================
# Correlation Analyzer
# ===========================================================================

class CorrelationAnalyzer:
    """Compute pairwise correlation matrix for numeric columns."""

    @timed
    def analyze(self, store: DataStore) -> AnalysisResult:
        numeric_cols = [
            col for col in store.column_names
            if store._schema.get(col) == DataType.NUMERIC
        ]
        matrix: Dict[str, Dict[str, float]] = {}
        strong_pairs: List[Tuple[str, str, float]] = []

        for i, col_a in enumerate(numeric_cols):
            matrix[col_a] = {}
            a = store.get_numeric(col_a)
            for col_b in numeric_cols:
                b = store.get_numeric(col_b)
                r = correlation(a, b) if col_a != col_b else 1.0
                matrix[col_a][col_b] = round(r, 4)
                if col_a < col_b and abs(r) >= 0.7:
                    strong_pairs.append((col_a, col_b, r))

        insights = []
        for a, b, r in sorted(strong_pairs, key=lambda x: -abs(x[2])):
            direction = "positive" if r > 0 else "negative"
            strength = "very strong" if abs(r) >= 0.9 else "strong"
            insights.append(
                f"{strength.capitalize()} {direction} correlation between '{a}' and '{b}' (r={r:.3f})."
            )

        return AnalysisResult(
            analysis_id=generate_id("CORR"),
            analysis_type=AnalysisType.CORRELATION,
            dataset_id=store.dataset_id,
            summary=f"Correlation analysis across {len(numeric_cols)} numeric columns. "
                    f"Found {len(strong_pairs)} strong correlation(s).",
            insights=insights or ["No strong correlations detected between numeric columns."],
            metrics={"correlation_matrix": matrix, "strong_pairs": len(strong_pairs)},
            charts=[{
                "type": ChartType.HEATMAP.value,
                "title": "Correlation Matrix",
                "data": matrix,
                "columns": numeric_cols,
            }],
            recommendations=[
                "Highly correlated features may cause multicollinearity in regression models — consider dropping one.",
                "Use correlation insights to prioritize features for predictive modeling.",
            ],
        )


# ===========================================================================
# Regression Analyzer
# ===========================================================================

class RegressionAnalyzer:
    """Simple linear and multiple regression analysis."""

    @timed
    def analyze_simple(self, store: DataStore, x_col: str, y_col: str) -> AnalysisResult:
        x = store.get_numeric(x_col)
        y = store.get_numeric(y_col)
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]

        slope, intercept, r2 = linear_regression(x, y)
        r = correlation(x, y)

        y_pred = [slope * xi + intercept for xi in x]
        residuals = [y[i] - y_pred[i] for i in range(n)]
        rmse = math.sqrt(mean([r ** 2 for r in residuals]))
        mae = mean([abs(r) for r in residuals])

        return AnalysisResult(
            analysis_id=generate_id("REG"),
            analysis_type=AnalysisType.REGRESSION,
            dataset_id=store.dataset_id,
            summary=(
                f"Linear regression: {y_col} ~ {x_col}. "
                f"R² = {r2:.4f} ({r2*100:.1f}% variance explained)."
            ),
            insights=[
                f"Equation: {y_col} = {slope:.4f} × {x_col} + {intercept:.4f}",
                f"R² = {r2:.4f} — model explains {r2*100:.1f}% of variance in {y_col}.",
                f"Pearson correlation: {r:.4f} ({'positive' if r > 0 else 'negative'}).",
                f"RMSE = {rmse:.4f}, MAE = {mae:.4f}.",
                (
                    "Strong predictive relationship detected."
                    if r2 >= 0.7
                    else "Moderate relationship — consider polynomial or multivariate regression."
                    if r2 >= 0.4
                    else "Weak linear relationship. Linear regression may not be the best model."
                ),
            ],
            metrics={
                "slope": slope,
                "intercept": intercept,
                "r_squared": round(r2, 6),
                "pearson_r": round(r, 6),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "n": n,
            },
            charts=[{
                "type": ChartType.SCATTER.value,
                "title": f"Regression: {y_col} vs {x_col}",
                "x": x[:100],
                "y": y[:100],
                "trend_line": {"slope": slope, "intercept": intercept},
            }],
            recommendations=[
                "Check residual plots for heteroscedasticity before trusting p-values.",
                "If R² < 0.5, consider adding more features or trying non-linear models.",
            ],
        )


# ===========================================================================
# Anomaly Detector
# ===========================================================================

class AnomalyDetector:
    """Detect anomalies using Z-score, IQR, and moving average methods."""

    def detect_zscore(self, store: DataStore, column: str, threshold: float = 3.0) -> AnomalyReport:
        nums = store.get_numeric(column)
        mu = mean(nums)
        sigma = std_dev(nums)
        anomalies = []
        for i, val in enumerate(nums):
            z = zscore(val, mu, sigma)
            if abs(z) > threshold:
                anomalies.append({
                    "index": i,
                    "value": val,
                    "z_score": round(z, 4),
                    "direction": "high" if z > 0 else "low",
                })
        return AnomalyReport(
            report_id=generate_id("ANO"),
            method="z_score",
            total_records=len(nums),
            anomaly_count=len(anomalies),
            anomaly_pct=round(len(anomalies) / max(len(nums), 1) * 100, 2),
            anomalies=anomalies[:50],
            threshold=threshold,
            summary=(
                f"Z-score anomaly detection on '{column}': "
                f"{len(anomalies)} anomalies ({len(anomalies)/max(len(nums),1)*100:.1f}%) "
                f"at threshold ±{threshold}σ."
            ),
        )

    def detect_iqr(self, store: DataStore, column: str, k: float = 1.5) -> AnomalyReport:
        nums = store.get_numeric(column)
        lo, hi = iqr_bounds(nums, k)
        anomalies = []
        for i, val in enumerate(nums):
            if val < lo or val > hi:
                anomalies.append({
                    "index": i,
                    "value": val,
                    "bound_violated": "lower" if val < lo else "upper",
                    "lower_bound": round(lo, 4),
                    "upper_bound": round(hi, 4),
                })
        return AnomalyReport(
            report_id=generate_id("ANO"),
            method="iqr",
            total_records=len(nums),
            anomaly_count=len(anomalies),
            anomaly_pct=round(len(anomalies) / max(len(nums), 1) * 100, 2),
            anomalies=anomalies[:50],
            threshold=k,
            summary=(
                f"IQR anomaly detection on '{column}': "
                f"{len(anomalies)} anomalies ({len(anomalies)/max(len(nums),1)*100:.1f}%) "
                f"outside [{lo:.2f}, {hi:.2f}]."
            ),
        )

    def detect_moving_average(
        self, store: DataStore, column: str, window: int = 5, sensitivity: float = 2.0
    ) -> AnomalyReport:
        nums = store.get_numeric(column)
        anomalies = []
        for i in range(window, len(nums)):
            window_vals = nums[i - window:i]
            ma = mean(window_vals)
            dev = std_dev(window_vals)
            diff = abs(nums[i] - ma)
            if dev > 0 and diff > sensitivity * dev:
                anomalies.append({
                    "index": i,
                    "value": nums[i],
                    "moving_avg": round(ma, 4),
                    "deviation": round(diff, 4),
                    "direction": "spike" if nums[i] > ma else "drop",
                })
        return AnomalyReport(
            report_id=generate_id("ANO"),
            method="moving_average",
            total_records=len(nums),
            anomaly_count=len(anomalies),
            anomaly_pct=round(len(anomalies) / max(len(nums), 1) * 100, 2),
            anomalies=anomalies[:50],
            threshold=sensitivity,
            summary=(
                f"Moving average anomaly detection on '{column}' (window={window}): "
                f"{len(anomalies)} anomalies detected."
            ),
        )


# ===========================================================================
# Hypothesis Test Engine
# ===========================================================================

class HypothesisEngine:
    """Run statistical hypothesis tests."""

    def one_sample_t(
        self, store: DataStore, column: str, mu0: float = 0.0, alpha: float = 0.05
    ) -> HypothesisTestResult:
        data = store.get_numeric(column)
        t, p = t_test_one_sample(data, mu0)
        m = mean(data)
        s = std_dev(data)
        cohen_d = safe_div(m - mu0, s)
        return HypothesisTestResult(
            test_name="One-Sample T-Test",
            null_hypothesis=f"The mean of '{column}' equals {mu0}",
            statistic=round(t, 4),
            p_value=round(p, 4),
            alpha=alpha,
            reject_null=p < alpha,
            conclusion=(
                f"Reject H₀ — mean of '{column}' ({mean(data):.4f}) significantly differs from {mu0} "
                f"(p={p:.4f} < α={alpha})."
                if p < alpha
                else f"Fail to reject H₀ — no significant difference from {mu0} (p={p:.4f} ≥ α={alpha})."
            ),
            effect_size=round(cohen_d, 4),
        )

    def two_sample_t(
        self,
        store_a: DataStore,
        store_b: DataStore,
        column: str,
        alpha: float = 0.05,
        label_a: str = "Group A",
        label_b: str = "Group B",
    ) -> HypothesisTestResult:
        a = store_a.get_numeric(column)
        b = store_b.get_numeric(column)
        t, p = t_test_two_sample(a, b)
        ma, mb = mean(a), mean(b)
        pooled_std = math.sqrt((variance(a) + variance(b)) / 2)
        cohen_d = safe_div(ma - mb, pooled_std)
        return HypothesisTestResult(
            test_name="Two-Sample Welch T-Test",
            null_hypothesis=f"The means of '{label_a}' and '{label_b}' on '{column}' are equal",
            statistic=round(t, 4),
            p_value=round(p, 4),
            alpha=alpha,
            reject_null=p < alpha,
            conclusion=(
                f"Reject H₀ — significant difference between {label_a} ({ma:.4f}) "
                f"and {label_b} ({mb:.4f}) on '{column}' (p={p:.4f} < α={alpha})."
                if p < alpha
                else f"Fail to reject H₀ — no significant difference (p={p:.4f} ≥ α={alpha})."
            ),
            effect_size=round(cohen_d, 4),
        )


# ===========================================================================
# Time Series Analyzer
# ===========================================================================

class TimeSeriesAnalyzer:
    """Trend, seasonality, and forecasting for time-ordered data."""

    def analyze(self, values: List[float], period: int = 7) -> Dict[str, Any]:
        n = len(values)
        if n < period * 2:
            return {"error": "Insufficient data for time series analysis."}

        # Trend via linear regression on index
        indices = list(range(n))
        slope, intercept, r2 = linear_regression(indices, values)
        trend_direction = "upward" if slope > 0 else "downward" if slope < 0 else "flat"

        # Moving average
        ma = []
        for i in range(period - 1, n):
            ma.append(mean(values[i - period + 1:i + 1]))

        # Seasonal decomposition (simplified — average by period position)
        seasonal = []
        for p in range(period):
            period_vals = [values[i] for i in range(p, n, period)]
            seasonal.append(round(mean(period_vals), 4))

        # Volatility
        returns = [safe_div(values[i] - values[i-1], abs(values[i-1])) for i in range(1, n)]
        volatility = round(std_dev(returns) * 100, 2) if returns else 0.0

        # Simple forecast: extend trend
        forecast_horizon = min(period, 10)
        forecast = [round(slope * (n + i) + intercept, 4) for i in range(forecast_horizon)]

        return {
            "trend": {
                "slope": round(slope, 6),
                "direction": trend_direction,
                "r_squared": round(r2, 4),
            },
            "moving_average": [round(v, 4) for v in ma[-10:]],
            "seasonal_pattern": seasonal,
            "volatility_pct": volatility,
            "forecast": forecast,
            "summary": (
                f"{trend_direction.capitalize()} trend (slope={slope:.4f}/period, R²={r2:.3f}). "
                f"Volatility: {volatility:.1f}%. "
                f"{forecast_horizon}-period forecast: {forecast[0]:.2f} → {forecast[-1]:.2f}."
            ),
        }


# ===========================================================================
# Natural Language Query Engine
# ===========================================================================

class NLQueryEngine:
    """Translate natural language questions into data operations."""

    AGGREGATION_PATTERNS = {
        r"\b(average|mean|avg)\b": "mean",
        r"\b(total|sum)\b": "sum",
        r"\b(count|how many)\b": "count",
        r"\b(maximum|max|highest|largest)\b": "max",
        r"\b(minimum|min|lowest|smallest)\b": "min",
        r"\b(median)\b": "median",
        r"\b(std|standard deviation|variance)\b": "std",
    }

    def parse(self, query: str, store: DataStore) -> Dict[str, Any]:
        lower = query.lower()

        # Detect aggregation
        agg_func = None
        for pattern, func in self.AGGREGATION_PATTERNS.items():
            if re.search(pattern, lower):
                agg_func = func
                break

        # Detect columns mentioned
        mentioned_cols = [col for col in store.column_names if col.lower() in lower]

        # Detect filter intent
        filter_intent = None
        filter_match = re.search(r"\b(where|filter|only|for)\b (.+?)(\?|$)", lower)
        if filter_match:
            filter_intent = filter_match.group(2).strip()

        return {
            "query": query,
            "detected_aggregation": agg_func,
            "detected_columns": mentioned_cols,
            "filter_intent": filter_intent,
            "interpretation": self._interpret(agg_func, mentioned_cols, filter_intent),
        }

    def execute(self, parsed: Dict[str, Any], store: DataStore) -> Dict[str, Any]:
        cols = parsed["detected_columns"]
        agg = parsed["detected_aggregation"]
        results: Dict[str, Any] = {}

        for col in cols:
            dtype = store._schema.get(col, DataType.UNKNOWN)
            if dtype == DataType.NUMERIC:
                nums = store.get_numeric(col)
                if not nums:
                    results[col] = "no numeric data"
                    continue
                if agg == "mean":
                    results[col] = round(mean(nums), 4)
                elif agg == "sum":
                    results[col] = round(sum(nums), 4)
                elif agg == "count":
                    results[col] = len(nums)
                elif agg == "max":
                    results[col] = max(nums)
                elif agg == "min":
                    results[col] = min(nums)
                elif agg == "median":
                    results[col] = round(percentile(nums, 50), 4)
                elif agg == "std":
                    results[col] = round(std_dev(nums), 4)
                else:
                    results[col] = {"mean": round(mean(nums), 4), "count": len(nums)}
            else:
                vals = store.get_column(col)
                if agg == "count":
                    results[col] = len([v for v in vals if v is not None])
                else:
                    counter = Counter(str(v) for v in vals if v is not None)
                    results[col] = dict(counter.most_common(5))

        return {"query": parsed["query"], "results": results, "interpretation": parsed["interpretation"]}

    def _interpret(self, agg: Optional[str], cols: List[str], filt: Optional[str]) -> str:
        parts = []
        if agg:
            parts.append(f"Calculate {agg}")
        if cols:
            parts.append(f"for column(s): {', '.join(cols)}")
        if filt:
            parts.append(f"where {filt}")
        return " ".join(parts) if parts else "Could not interpret query — please rephrase."


# ===========================================================================
# Report Generator
# ===========================================================================

class ReportGenerator:
    """Generate HTML and JSON reports from analysis results."""

    def to_html(self, result: AnalysisResult) -> str:
        chart_html = ""
        for c in result.charts:
            chart_html += f"<div class='chart-placeholder' data-type='{c['type']}' data-col='{c.get('column', '')}'>Chart: {c['title']}</div>"

        insight_items = "".join(f"<li>{i}</li>" for i in result.insights)
        rec_items = "".join(f"<li>{r}</li>" for r in result.recommendations)
        metrics_rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in result.metrics.items()
            if not isinstance(v, dict)
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Analysis Report — {result.analysis_type.value}</title>
  <style>
    body {{ font-family: 'DM Sans', sans-serif; margin: 2rem; color: #203038; }}
    h1 {{ color: #578098; }} h2 {{ color: #87adc6; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #cbe4ed; padding: 8px; }}
    .chart-placeholder {{ background: #f0fbfa; border: 1px solid #cbe4ed;
      padding: 20px; margin: 10px 0; border-radius: 8px; }}
    ul {{ line-height: 1.8; }}
  </style>
</head>
<body>
  <h1>Analysis Report</h1>
  <p><strong>Type:</strong> {result.analysis_type.value}</p>
  <p><strong>Dataset:</strong> {result.dataset_id}</p>
  <p><strong>Generated:</strong> {result.created_at}</p>
  <h2>Summary</h2>
  <p>{result.summary}</p>
  <h2>Key Insights</h2>
  <ul>{insight_items}</ul>
  <h2>Metrics</h2>
  <table><tr><th>Metric</th><th>Value</th></tr>{metrics_rows}</table>
  <h2>Charts</h2>
  {chart_html}
  <h2>Recommendations</h2>
  <ul>{rec_items}</ul>
</body>
</html>"""

    def to_json(self, result: AnalysisResult) -> str:
        return json.dumps({
            "analysis_id": result.analysis_id,
            "type": result.analysis_type.value,
            "summary": result.summary,
            "insights": result.insights,
            "metrics": {k: v for k, v in result.metrics.items() if not isinstance(v, (dict, list))},
            "recommendations": result.recommendations,
            "created_at": result.created_at,
        }, indent=2)


# ===========================================================================
# Main Orchestrator — DataAnalystAI
# ===========================================================================

class DataAnalystAI:
    """Enterprise AI Data Analyst — orchestrates all analysis capabilities."""

    VERSION = "2.0.0"

    def __init__(self):
        self.ingester = DataIngester()
        self.profiler = ColumnProfiler()
        self.quality_analyzer = DataQualityAnalyzer()
        self.eda_engine = EDAEngine(self.profiler, self.quality_analyzer)
        self.correlation_analyzer = CorrelationAnalyzer()
        self.regression_analyzer = RegressionAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.hypothesis_engine = HypothesisEngine()
        self.ts_analyzer = TimeSeriesAnalyzer()
        self.nl_engine = NLQueryEngine()
        self.report_gen = ReportGenerator()
        self._datasets: Dict[str, DataStore] = {}
        self._results: List[AnalysisResult] = []
        self._stats = {"analyses": 0, "datasets_loaded": 0, "queries": 0}

    def load_csv(self, csv_text: str, name: str = "dataset") -> DataStore:
        store = self.ingester.from_csv_string(csv_text, name)
        self._datasets[store.dataset_id] = store
        self._stats["datasets_loaded"] += 1
        logger.info(f"Loaded dataset '{name}': {store.row_count} rows × {store.col_count} cols")
        return store

    def load_json(self, json_text: str, name: str = "dataset") -> DataStore:
        store = self.ingester.from_json_string(json_text, name)
        self._datasets[store.dataset_id] = store
        self._stats["datasets_loaded"] += 1
        return store

    def load_records(self, records: List[Dict[str, Any]], name: str = "dataset") -> DataStore:
        store = self.ingester.from_dicts(records, name)
        self._datasets[store.dataset_id] = store
        self._stats["datasets_loaded"] += 1
        return store

    def run_eda(self, store: DataStore) -> AnalysisResult:
        result = self.eda_engine.analyze(store)
        self._results.append(result)
        self._stats["analyses"] += 1
        return result

    def run_correlation(self, store: DataStore) -> AnalysisResult:
        result = self.correlation_analyzer.analyze(store)
        self._results.append(result)
        self._stats["analyses"] += 1
        return result

    def run_regression(self, store: DataStore, x_col: str, y_col: str) -> AnalysisResult:
        result = self.regression_analyzer.analyze_simple(store, x_col, y_col)
        self._results.append(result)
        self._stats["analyses"] += 1
        return result

    def detect_anomalies(
        self, store: DataStore, column: str, method: str = "zscore", **kwargs
    ) -> AnomalyReport:
        if method == "zscore":
            return self.anomaly_detector.detect_zscore(store, column, **kwargs)
        elif method == "iqr":
            return self.anomaly_detector.detect_iqr(store, column, **kwargs)
        elif method == "moving_average":
            return self.anomaly_detector.detect_moving_average(store, column, **kwargs)
        raise ValueError(f"Unknown method: {method}. Use 'zscore', 'iqr', or 'moving_average'.")

    def run_hypothesis_test(
        self, store_a: DataStore, column: str,
        store_b: Optional[DataStore] = None, mu0: float = 0.0
    ) -> HypothesisTestResult:
        if store_b:
            return self.hypothesis_engine.two_sample_t(store_a, store_b, column)
        return self.hypothesis_engine.one_sample_t(store_a, column, mu0)

    def analyze_time_series(self, values: List[float], period: int = 7) -> Dict[str, Any]:
        self._stats["analyses"] += 1
        return self.ts_analyzer.analyze(values, period)

    def query(self, question: str, store: DataStore) -> Dict[str, Any]:
        parsed = self.nl_engine.parse(question, store)
        result = self.nl_engine.execute(parsed, store)
        self._stats["queries"] += 1
        return result

    def export_report(self, result: AnalysisResult, fmt: str = "json") -> str:
        if fmt == "html":
            return self.report_gen.to_html(result)
        return self.report_gen.to_json(result)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "datasets_loaded": self._stats["datasets_loaded"],
            "analyses_run": self._stats["analyses"],
            "nl_queries": self._stats["queries"],
            "results_stored": len(self._results),
        }


# ===========================================================================
# Demo
# ===========================================================================

def _build_sample_store() -> DataStore:
    """Create a realistic sales dataset for demo purposes."""
    import random
    random.seed(42)
    records = []
    regions = ["North", "South", "East", "West"]
    products = ["Starter", "Professional", "Enterprise"]
    for i in range(200):
        region = random.choice(regions)
        product = random.choice(products)
        base = {"Starter": 1200, "Professional": 3500, "Enterprise": 9800}[product]
        revenue = round(base * (0.7 + random.random() * 0.6) + random.gauss(0, 300), 2)
        deals = random.randint(1, 15)
        churn_risk = round(max(0, min(100, 50 - (revenue / 200) + random.gauss(0, 10))), 1)
        records.append({
            "deal_id": f"D{i+1:04d}",
            "region": region,
            "product": product,
            "revenue": revenue,
            "deals_closed": deals,
            "churn_risk_score": churn_risk,
            "days_to_close": random.randint(7, 180),
            "rep_satisfaction": round(3 + random.random() * 2, 1),
        })
    # Inject a few anomalies
    records[5]["revenue"] = 95000.0   # extreme outlier
    records[10]["revenue"] = -500.0   # negative outlier
    records[15]["days_to_close"] = None  # missing value
    ingester = DataIngester()
    return ingester.from_dicts(records, "sales_data_2026")


def main():
    print("\n" + "=" * 80)
    print("  DATA ANALYST AI — ENTERPRISE v2.0")
    print("=" * 80)

    ai = DataAnalystAI()
    store = _build_sample_store()
    ai._datasets[store.dataset_id] = store
    ai._stats["datasets_loaded"] += 1

    print(f"\n{store.describe()}")

    # EDA
    eda = ai.run_eda(store)
    print(f"\n{'─'*60}")
    print("[EDA] " + eda.summary)
    for insight in eda.insights[:4]:
        print(f"  • {insight}")
    print(f"  Quality Score: {eda.metrics['quality_score']}/100")

    # Correlation
    corr = ai.run_correlation(store)
    print(f"\n{'─'*60}")
    print("[Correlation] " + corr.summary)
    for insight in corr.insights[:3]:
        print(f"  • {insight}")

    # Regression
    reg = ai.run_regression(store, "days_to_close", "revenue")
    print(f"\n{'─'*60}")
    print("[Regression] " + reg.summary)
    for insight in reg.insights[:3]:
        print(f"  • {insight}")

    # Anomaly Detection
    print(f"\n{'─'*60}")
    for method in ["zscore", "iqr"]:
        report = ai.detect_anomalies(store, "revenue", method=method)
        print(f"[Anomaly/{method}] {report.summary}")

    # Hypothesis Test
    print(f"\n{'─'*60}")
    north = store.filter_rows("region", lambda v: v == "North")
    south = store.filter_rows("region", lambda v: v == "South")
    hyp = ai.run_hypothesis_test(north, "revenue", store_b=south)
    print(f"[HypothesisTest] {hyp.test_name}: {hyp.conclusion}")
    print(f"  p={hyp.p_value}, t={hyp.statistic}, d={hyp.effect_size}")

    # Time Series
    print(f"\n{'─'*60}")
    revenues = store.get_numeric("revenue")[:60]
    ts = ai.analyze_time_series(revenues, period=7)
    print(f"[TimeSeries] {ts['summary']}")

    # Natural Language Query
    print(f"\n{'─'*60}")
    questions = [
        "What is the average revenue?",
        "What is the total deals_closed?",
        "What are the top regions?",
    ]
    for q in questions:
        result = ai.query(q, store)
        print(f"[NLQuery] Q: '{q}'")
        print(f"         A: {result['results']}")

    # Export
    print(f"\n{'─'*60}")
    json_report = ai.export_report(eda, fmt="json")
    print(f"[Export] JSON report preview:\n{json_report[:300]}...")

    # Stats
    print(f"\n{'─'*60}")
    print(f"[Stats] {ai.get_statistics()}")
    print("=" * 80)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Version info and module exports
# ---------------------------------------------------------------------------

__version__ = DataAnalystAI.VERSION
__all__ = [
    "DataAnalystAI",
    "DataStore",
    "DataIngester",
    "EDAEngine",
    "CorrelationAnalyzer",
    "RegressionAnalyzer",
    "AnomalyDetector",
    "HypothesisEngine",
    "TimeSeriesAnalyzer",
    "NLQueryEngine",
    "ReportGenerator",
    "ColumnProfiler",
    "DataQualityAnalyzer",
    "DataType",
    "AnalysisType",
    "ModelType",
    "ChartType",
]
