# Copyright ©2022-2023. The Regents of the University of California
# (Regents). All Rights Reserved.
#
# SPDX-License-Identifier: BSD-2-Clause

"""
Map CDR feature column names to the legacy (pre-2025) CIDER export convention.

Used so new parquet/CSV outputs align with historical files such as
``all_features_with_hhids.parquet`` (stat suffix last; no doubled pivot tokens).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

import dask.dataframe as dd
import pandas as pd


def cdr_column_to_legacy(name: str) -> str:
    """Return the legacy column name for one CDR feature column (``name`` unchanged if N/A)."""
    if name in ("name", "caller_id"):
        return name

    c = name

    # Remove doubled tokens from pivot(value == indicator) or value name echoes
    c = re.sub(r"^active_days_active_days_", "active_days_", c)
    c = re.sub(r"^number_of_contacts_number_of_contacts_", "number_of_contacts_", c)
    c = re.sub(r"^percent_nocturnal_percent_nocturnal_", "percent_nocturnal_", c)
    c = re.sub(
        r"^percent_initiated_conversations_percent_initiated_conversations_",
        "percent_initiated_conversations_",
        c,
    )
    c = re.sub(
        r"^percent_initiated_interactions_percent_initiated_interactions_",
        "percent_initiated_interactions_",
        c,
    )
    c = re.sub(r"^entropy_of_contacts_entropy_", "entropy_of_contacts_", c)
    c = re.sub(
        r"^interactions_per_contact_interactions_per_contact_",
        "interactions_per_contact_",
        c,
    )
    c = re.sub(r"^interevent_time_interevent_time_", "interevent_time_", c)
    c = re.sub(
        r"^percent_pareto_interactions_percent_pareto_interactions_",
        "percent_pareto_interactions_",
        c,
    )
    c = re.sub(
        r"^percent_pareto_durations_percent_pareto_durations_",
        "percent_pareto_durations_",
        c,
    )
    c = re.sub(r"^response_rate_text_response_rate_text_", "response_rate_text_", c)

    c = re.sub(r"^number_of_interactions_n_", "number_of_interactions_", c)

    c = re.sub(r"^number_of_antennas_n_antennas_", "number_of_antennas_", c)
    c = re.sub(r"^entropy_of_antennas_entropy_", "entropy_of_antennas_", c)
    c = re.sub(r"^frequent_antennas_pareto_antennas_", "frequent_antennas_", c)
    c = re.sub(r"^radius_of_gyration_r_", "radius_of_gyration_", c)

    c = re.sub(r"^percent_pareto_interactions_pareto_", "percent_pareto_interactions_", c)
    c = re.sub(r"^percent_pareto_durations_pareto_", "percent_pareto_durations_", c)

    # Legacy percent_at_home used a single scalar per (weekday, daytime); only mean exists today
    c = re.sub(r"^percent_at_home_mean_", "percent_at_home_", c)

    # --- Move summary-stat token to the end (legacy order) ---

    m = re.match(
        r"^(balance_of_contacts|interactions_per_contact|interevent_time)_"
        r"(mean|std|median|skewness|kurtosis|min|max)_"
        r"(allweek|weekday|weekend)_(allday|day|night)_(call|text)$",
        c,
    )
    if m:
        base, stat, aw, dn, tx = m.groups()
        return f"{base}_{aw}_{dn}_{tx}_{stat}"

    m = re.match(
        r"^(call_duration|response_delay_text)_"
        r"(mean|std|median|skewness|kurtosis|min|max)_"
        r"(allweek|weekday|weekend)_(allday|day|night)_(call|text)$",
        c,
    )
    if m:
        base, stat, aw, dn, tx = m.groups()
        return f"{base}_{aw}_{dn}_{tx}_{stat}"

    m = re.match(
        r"^(response_delay_text)_"
        r"(mean|std|median|skewness|kurtosis|min|max)_"
        r"(allweek|weekday|weekend)_(allday|day|night)$",
        c,
    )
    if m:
        base, stat, aw, dn = m.groups()
        return f"{base}_{aw}_{dn}_{stat}"

    m = re.match(
        r"^percent_at_home_"
        r"(mean|std|median|skewness|kurtosis|min|max)_"
        r"(allweek|weekday|weekend)_(allday|day|night)$",
        c,
    )
    if m:
        stat, aw, dn = m.groups()
        if stat == "mean":
            return f"percent_at_home_{aw}_{dn}"
        return f"percent_at_home_{aw}_{dn}_{stat}"

    return c


def legacy_rename_mapping(columns: List[str]) -> Dict[str, str]:
    """Build {old: new} only for columns whose legacy name differs."""
    out: Dict[str, str] = {}
    for c in columns:
        new = cdr_column_to_legacy(c)
        if new != c:
            out[c] = new
    targets = list(out.values())
    dup = [k for k, v in Counter(targets).items() if v > 1]
    if dup:
        raise ValueError(
            "Legacy CDR rename would create duplicate column names "
            f"(examples: {dup[:8]}). Report this as a CIDER bug."
        )
    return out


def rename_cdr_features_to_legacy(df: dd.DataFrame) -> dd.DataFrame:
    """Rename CDR feature columns on a Dask (or pandas) DataFrame to legacy names."""
    mapping = legacy_rename_mapping(list(df.columns))
    if not mapping:
        return df
    return df.rename(columns=mapping)


def rename_cdr_features_to_legacy_pandas(df: pd.DataFrame) -> pd.DataFrame:
    mapping = legacy_rename_mapping(list(df.columns))
    if not mapping:
        return df
    return df.rename(columns=mapping)
