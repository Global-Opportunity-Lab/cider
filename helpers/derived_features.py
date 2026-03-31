"""
Derived features computed by comparing a baseline window against a shock window,
or by fitting trends across multiple ordered windows.

Baseline/shock functions take two Pandas DataFrames keyed by 'name' and return a
DataFrame with 'name' plus derived columns. Callers provide window lengths in days
where required for rate normalisation.

compute_trend_features() takes an ordered list of DataFrames (one per time window)
and fits per-person slopes across all numeric features, designed for slow-onset shocks.

Column naming conventions:
  __delta              shock minus baseline (signed; negative = dropped)
  __rate_ratio         (shock_rate) / (baseline_rate); >1 means elevated in shock window
  __churnable          may be confounded by undetected SIM/phone-number churn
  __routine_collapse   drop in abs(weekday − weekend) difference; negative = lost weekly rhythm
  __slope              OLS slope across ordered windows (units: change per day)
  __sign_consistency   fraction of consecutive window-pairs moving in the slope direction
"""
import numpy as np
import pandas as pd
from datetime import date


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _common_cols(baseline_df: pd.DataFrame, shock_df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in shock_df.columns if c.startswith(prefix) and c in baseline_df.columns]


def _delta(baseline_df: pd.DataFrame, shock_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return shock - baseline for each column, merged on 'name'."""
    b = baseline_df[['name'] + cols].rename(columns={c: f'{c}__b' for c in cols})
    s = shock_df[['name'] + cols].rename(columns={c: f'{c}__s' for c in cols})
    m = s.merge(b, on='name', how='inner')
    result = m[['name']].copy()
    for c in cols:
        result[f'{c}__delta'] = m[f'{c}__s'] - m[f'{c}__b']
    return result


def _rate_ratio(
    baseline_df: pd.DataFrame,
    shock_df: pd.DataFrame,
    cols: list[str],
    baseline_days: int,
    shock_days: int,
    suffix: str = '__rate_ratio',
) -> pd.DataFrame:
    """Return (shock/shock_days) / (baseline/baseline_days) for each column."""
    b = baseline_df[['name'] + cols].copy()
    s = shock_df[['name'] + cols].copy()
    for c in cols:
        b[c] = b[c] / baseline_days
        s[c] = s[c] / shock_days
    m = s.rename(columns={c: f'{c}__s' for c in cols}).merge(
        b.rename(columns={c: f'{c}__b' for c in cols}),
        on='name', how='inner'
    )
    result = m[['name']].copy()
    for c in cols:
        result[f'{c}{suffix}'] = m[f'{c}__s'] / m[f'{c}__b'].replace(0, float('nan'))
    return result


# ---------------------------------------------------------------------------
# CDR-derived features
# ---------------------------------------------------------------------------

def direction_asymmetry_features(
    baseline_df: pd.DataFrame,
    shock_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Feature #3: Change in outgoing interaction and conversation fraction.

    Uses percent_initiated_interactions_* and percent_initiated_conversations_* which
    CIDER already produces. A negative delta means the person is receiving relatively
    more calls — consistent with social network mobilisation after a death or illness.
    """
    cols = (
        _common_cols(baseline_df, shock_df, 'percent_initiated_interactions_')
        + _common_cols(baseline_df, shock_df, 'percent_initiated_conversations_')
    )
    return _delta(baseline_df, shock_df, cols)


def silence_features(
    baseline_df: pd.DataFrame,
    shock_df: pd.DataFrame,
    baseline_days: int,
    shock_days: int,
) -> pd.DataFrame:
    """
    Feature #10: Drop in active-day rate from baseline to shock window.

    Active days are normalised by window length before differencing so that a short
    shock window is directly comparable to a long baseline window.

    Columns are suffixed _churnable: undetected SIM churn produces false-positive
    silence signals that are indistinguishable from genuine inactivity.
    """
    cols = _common_cols(baseline_df, shock_df, 'active_days_')
    b = baseline_df[['name'] + cols].copy()
    s = shock_df[['name'] + cols].copy()
    for c in cols:
        b[c] = b[c] / baseline_days
        s[c] = s[c] / shock_days
    m = s.rename(columns={c: f'{c}__s' for c in cols}).merge(
        b.rename(columns={c: f'{c}__b' for c in cols}),
        on='name', how='inner'
    )
    result = m[['name']].copy()
    for c in cols:
        result[f'{c}__active_rate_drop_churnable'] = m[f'{c}__b'] - m[f'{c}__s']
    return result


def new_contact_mobilization_features(
    baseline_df: pd.DataFrame,
    shock_df: pd.DataFrame,
    baseline_days: int,
    shock_days: int,
) -> pd.DataFrame:
    """
    Feature #6 (approximate): Ratio of per-day unique contact rate in shock vs baseline.

    Values > 1 indicate contact mobilisation. Cannot distinguish truly new contacts
    from returning ones without raw CDR contact sets.
    """
    cols = _common_cols(baseline_df, shock_df, 'number_of_contacts_')
    return _rate_ratio(baseline_df, shock_df, cols, baseline_days, shock_days, '__contact_rate_ratio')


def temporal_disruption_features(
    baseline_df: pd.DataFrame,
    shock_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Feature #2 (proxy): Change in nocturnal call fraction from baseline to shock window.

    Proxy for full intra-day temporal disruption (KL-divergence of hourly distributions),
    which requires per-hour CDR counts not currently produced by CIDER. Elevated nocturnal
    fraction signals sleep disruption consistent with acute crisis.
    """
    cols = _common_cols(baseline_df, shock_df, 'percent_nocturnal_')
    return _delta(baseline_df, shock_df, cols)


def communication_volume_features(
    baseline_df: pd.DataFrame,
    shock_df: pd.DataFrame,
    baseline_days: int,
    shock_days: int,
) -> pd.DataFrame:
    """
    Change in overall interaction rate (all-direction calls and texts).

    Captures both crisis-mobilisation spikes and hospitalisation/death crashes, which
    are opposite patterns that direction asymmetry and contact mobilisation alone cannot
    fully separate. Uses number_of_interactions_alldir_* columns.
    """
    cols = _common_cols(baseline_df, shock_df, 'number_of_interactions_alldir_')
    return _rate_ratio(baseline_df, shock_df, cols, baseline_days, shock_days, '__volume_rate_ratio')


def mobility_disruption_features(
    baseline_df: pd.DataFrame,
    shock_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Change in spatial mobility and home-dwell time.

    number_of_antennas delta: negative = contraction (illness), positive = expansion (displacement).
    percent_at_home delta:    positive = more time at home (illness), negative = displacement.
    These two features move in opposite directions for illness vs flood/cyclone shocks,
    making their combination more informative than either alone.
    """
    cols = (
        _common_cols(baseline_df, shock_df, 'number_of_antennas_')
        + _common_cols(baseline_df, shock_df, 'percent_at_home_')
    )
    return _delta(baseline_df, shock_df, cols)


def international_call_surge_features(
    baseline_df: pd.DataFrame,
    shock_df: pd.DataFrame,
    baseline_days: int,
    shock_days: int,
) -> pd.DataFrame:
    """
    Change in international contact rate.

    Contacting diaspora relatives is a strong signal for death or serious illness.
    Uses international_*__recipient_id__count and __nunique columns produced by
    CIDER's international featurizer.
    """
    cols = (
        _common_cols(baseline_df, shock_df, 'international_')
    )
    # Filter to count/nunique columns only — amount columns don't apply here
    cols = [c for c in cols if c.endswith('__count') or c.endswith('__nunique')]
    return _rate_ratio(baseline_df, shock_df, cols, baseline_days, shock_days, '__intl_rate_ratio')


def balance_of_contacts_shift_features(
    baseline_df: pd.DataFrame,
    shock_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Change in dyadic reciprocity score (balance of contacts).

    CIDER computes balance_of_contacts as the mean fraction of a dyad's interactions
    initiated by the ego. A shift toward negative means the network is reaching out
    *to* the person — consistent with condolence and support calling after a shock.
    Uses the mean statistic across all window/txn-type combinations.
    """
    cols = [
        c for c in _common_cols(baseline_df, shock_df, 'balance_of_contacts_')
        if c.endswith('_mean')
    ]
    return _delta(baseline_df, shock_df, cols)


# ---------------------------------------------------------------------------
# Recharge-derived features
# ---------------------------------------------------------------------------

def recharge_stress_features(
    baseline_df: pd.DataFrame,
    shock_df: pd.DataFrame,
    baseline_days: int,
    shock_days: int,
) -> pd.DataFrame:
    """
    Financial stress fingerprint from recharge behaviour.

    Two complementary signals:
    - recharges_mean__delta: negative = smaller individual recharge amounts (micro-recharge behaviour)
    - recharges_count__rate_ratio: >1 = more frequent recharges (topping up in small increments)

    Together they distinguish financial stress (small + frequent) from reduced usage
    (small + infrequent).
    """
    mean_cols = [c for c in _common_cols(baseline_df, shock_df, 'recharges_') if c.endswith('_mean')]
    count_cols = [c for c in _common_cols(baseline_df, shock_df, 'recharges_') if c.endswith('_count')]

    parts = []
    if mean_cols:
        parts.append(_delta(baseline_df, shock_df, mean_cols))
    if count_cols:
        parts.append(_rate_ratio(baseline_df, shock_df, count_cols, baseline_days, shock_days))
    if not parts:
        return baseline_df[['name']].copy().iloc[:0]

    result = parts[0]
    for part in parts[1:]:
        result = result.merge(part, on='name', how='outer')
    return result


# ---------------------------------------------------------------------------
# Mobile data features
# ---------------------------------------------------------------------------

def mobile_data_features(
    baseline_df: pd.DataFrame,
    shock_df: pd.DataFrame,
    baseline_days: int,
    shock_days: int,
) -> pd.DataFrame:
    """
    Change in mobile data usage.

    mobiledata_total_volume rate ratio: >1 = spike (information-seeking, online help);
    <1 = drop (can't afford data after financial shock).
    mobiledata_num_days rate ratio: captures whether data usage days changed, independent
    of volume per session.
    """
    volume_cols = _common_cols(baseline_df, shock_df, 'mobiledata_total_volume')
    days_cols = _common_cols(baseline_df, shock_df, 'mobiledata_num_days')
    mean_cols = _common_cols(baseline_df, shock_df, 'mobiledata_mean_daily_volume')
    cols = volume_cols + days_cols + mean_cols
    if not cols:
        return baseline_df[['name']].copy().iloc[:0]
    return _rate_ratio(baseline_df, shock_df, cols, baseline_days, shock_days)


# ---------------------------------------------------------------------------
# Mobile money features
# ---------------------------------------------------------------------------

def mobile_money_features(
    baseline_df: pd.DataFrame,
    shock_df: pd.DataFrame,
    baseline_days: int,
    shock_days: int,
) -> pd.DataFrame:
    """
    Change in mobile money behaviour.

    Three signals:
    - Outgoing amount rate ratio: >1 = elevated spending (hospital bills, crisis expenses)
    - Outgoing/incoming amount ratio delta: increase = cashout pressure (drawing down savings)
    - Outgoing transaction rate ratio: >1 = more frequent payments (financial scrambling)
    """
    result_parts = []

    # Outgoing volume and transaction rate
    out_amount_cols = [
        c for c in _common_cols(baseline_df, shock_df, 'mobilemoney_outgoing_')
        if 'amount' in c
    ]
    out_count_cols = [
        c for c in _common_cols(baseline_df, shock_df, 'mobilemoney_outgoing_')
        if c.endswith('_count')
    ]
    if out_amount_cols:
        result_parts.append(
            _rate_ratio(baseline_df, shock_df, out_amount_cols, baseline_days, shock_days)
        )
    if out_count_cols:
        result_parts.append(
            _rate_ratio(baseline_df, shock_df, out_count_cols, baseline_days, shock_days)
        )

    # Cashout pressure: outgoing/incoming amount ratio change
    out_mean_col = 'mobilemoney_outgoing_all_amount_mean'
    in_mean_col = 'mobilemoney_incoming_all_amount_mean'
    if all(c in baseline_df.columns and c in shock_df.columns for c in [out_mean_col, in_mean_col]):
        for df_label, df in [('b', baseline_df), ('s', shock_df)]:
            pass  # computed inline below
        b = baseline_df[['name', out_mean_col, in_mean_col]].copy()
        s = shock_df[['name', out_mean_col, in_mean_col]].copy()
        b['cashout_ratio__b'] = b[out_mean_col] / b[in_mean_col].replace(0, float('nan'))
        s['cashout_ratio__s'] = s[out_mean_col] / s[in_mean_col].replace(0, float('nan'))
        m = s[['name', 'cashout_ratio__s']].merge(b[['name', 'cashout_ratio__b']], on='name', how='inner')
        cashout = m[['name']].copy()
        cashout['mobilemoney_cashout_ratio__delta'] = m['cashout_ratio__s'] - m['cashout_ratio__b']
        result_parts.append(cashout)

    if not result_parts:
        return baseline_df[['name']].copy().iloc[:0]

    result = result_parts[0]
    for part in result_parts[1:]:
        result = result.merge(part, on='name', how='outer')
    return result


# ---------------------------------------------------------------------------
# Additional acute-shock features
# ---------------------------------------------------------------------------

def network_concentration_features(
    baseline_df: pd.DataFrame,
    shock_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Change in Pareto concentration — fraction of interactions going to top contacts.

    CIDER's percent_pareto_interactions_* measures the share of interactions accounted
    for by the minimum number of contacts that make up 80% of activity. A positive delta
    means calls are concentrating on fewer people — the "network collapse to core" pattern
    consistent with acute crisis (calling the same doctor or close relative repeatedly).

    Also includes percent_pareto_durations_* which captures the same collapse in call-time
    rather than interaction count.
    """
    cols = (
        _common_cols(baseline_df, shock_df, 'percent_pareto_interactions_')
        + _common_cols(baseline_df, shock_df, 'percent_pareto_durations_')
    )
    return _delta(baseline_df, shock_df, cols)


def response_rate_features(
    baseline_df: pd.DataFrame,
    shock_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Change in text response rate.

    CIDER's response_rate_text_* is the fraction of received texts that get a reply.
    A sudden drop signals unavailability — hospitalization, acute grief, or death —
    and is distinct from the silence features (which measure outgoing activity) because
    it captures *reactive* behaviour: the person stops responding even when reached.
    """
    cols = _common_cols(baseline_df, shock_df, 'response_rate_text_')
    return _delta(baseline_df, shock_df, cols)


def routine_collapse_features(
    baseline_df: pd.DataFrame,
    shock_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Loss of weekday/weekend behavioural differentiation.

    For each feature with both _weekday_ and _weekend_ variants, compute
    abs(weekday − weekend) as "routine strength" in each window. A negative delta
    means the weekly rhythm has collapsed — a signal of acute disruption that is
    independent of the absolute activity level. Captures routine loss even when total
    volume is unchanged (e.g., equally reduced on both weekdays and weekends).
    """
    # Find columns that have a _weekend_ counterpart
    weekday_cols = [c for c in baseline_df.columns if '_weekday_' in c and c in shock_df.columns]
    pairs = []
    for wday_col in weekday_cols:
        wend_col = wday_col.replace('_weekday_', '_weekend_')
        if wend_col in baseline_df.columns and wend_col in shock_df.columns:
            pairs.append((wday_col, wend_col))

    if not pairs:
        return baseline_df[['name']].copy().iloc[:0]

    result = baseline_df[['name']].merge(shock_df[['name']], on='name', how='inner').drop_duplicates()
    for wday_col, wend_col in pairs:
        base_name = wday_col.replace('_weekday_', '_')
        col_name = f'{base_name}_routine_collapse'
        baseline_strength = (baseline_df[wday_col] - baseline_df[wend_col]).abs()
        shock_strength = (shock_df[wday_col] - shock_df[wend_col]).abs()

        b = baseline_df[['name']].copy()
        b['strength'] = baseline_strength.values
        s = shock_df[['name']].copy()
        s['strength'] = shock_strength.values
        m = s.merge(b, on='name', how='inner', suffixes=('__s', '__b'))
        delta = m[['name']].copy()
        delta[col_name] = m['strength__s'] - m['strength__b']
        result = result.merge(delta, on='name', how='left')

    return result


# ---------------------------------------------------------------------------
# Slow-onset shock features: multi-window trend fitting
# ---------------------------------------------------------------------------

def compute_trend_features(
    window_dfs: list[pd.DataFrame],
    window_dates: list[date],
    min_windows: int = 3,
) -> pd.DataFrame:
    """
    Fit per-person OLS slopes across an ordered sequence of feature windows.

    Designed for slow-onset shocks (drought, chronic illness, gradual impoverishment)
    where the signal is a persistent monotonic trend rather than a sudden jump.

    Args:
        window_dfs: Feature DataFrames ordered oldest to newest, each keyed by 'name'.
        window_dates: Midpoint date of each window (same length as window_dfs).
        min_windows: Minimum number of windows a person must appear in to receive
                     trend features; others get NaN.

    Returns:
        DataFrame keyed by 'name' with two columns per feature:
          {feature}__slope            OLS slope (change per day)
          {feature}__sign_consistency fraction of consecutive window-pairs that move
                                      in the same direction as the overall slope;
                                      1.0 = perfectly monotonic, 0.5 = random walk
    """
    if len(window_dfs) < 2:
        raise ValueError('At least 2 windows required to compute trends.')
    if len(window_dfs) != len(window_dates):
        raise ValueError('window_dfs and window_dates must have the same length.')

    # Convert dates to numeric (days since first window)
    t0 = window_dates[0]
    days = np.array([(d - t0).days for d in window_dates], dtype=float)

    # Identify numeric feature columns (exclude 'name')
    feature_cols = [
        c for c in window_dfs[0].columns
        if c != 'name' and pd.api.types.is_numeric_dtype(window_dfs[0][c])
    ]

    # Stack all windows: person × (window_index, feature_value)
    tagged = []
    for i, df in enumerate(window_dfs):
        t = df[['name'] + feature_cols].copy()
        t['__t'] = days[i]
        tagged.append(t)
    stacked = pd.concat(tagged, ignore_index=True)

    results = []
    for name, grp in stacked.groupby('name'):
        row: dict = {'name': name}
        n = len(grp)
        if n < min_windows:
            for c in feature_cols:
                row[f'{c}__slope'] = float('nan')
                row[f'{c}__sign_consistency'] = float('nan')
            results.append(row)
            continue

        t_vals = grp['__t'].values
        for c in feature_cols:
            y = grp[c].values
            valid = ~np.isnan(y)
            if valid.sum() < min_windows:
                row[f'{c}__slope'] = float('nan')
                row[f'{c}__sign_consistency'] = float('nan')
                continue

            slope = float(np.polyfit(t_vals[valid], y[valid], 1)[0])
            row[f'{c}__slope'] = slope

            # Sign consistency: fraction of consecutive pairs moving in slope direction
            y_ordered = y[valid][np.argsort(t_vals[valid])]
            diffs = np.diff(y_ordered)
            if len(diffs) == 0 or slope == 0:
                row[f'{c}__sign_consistency'] = float('nan')
            else:
                expected_sign = np.sign(slope)
                row[f'{c}__sign_consistency'] = float(np.mean(np.sign(diffs) == expected_sign))

        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def compute_all_derived_features(
    baseline_df: pd.DataFrame,
    shock_df: pd.DataFrame,
    baseline_days: int,
    shock_days: int,
) -> pd.DataFrame:
    """
    Compute all derived shock-detection features and join into a single DataFrame.

    Args:
        baseline_df: Person-level features from the baseline window.
        shock_df: Person-level features from the shock window.
        baseline_days: Length of the baseline window in days (for rate normalisation).
        shock_days: Length of the shock window in days (for rate normalisation).

    Returns:
        DataFrame keyed by 'name' with all derived feature columns. Only persons
        present in both windows are included.
    """
    parts = [
        direction_asymmetry_features(baseline_df, shock_df),
        silence_features(baseline_df, shock_df, baseline_days, shock_days),
        new_contact_mobilization_features(baseline_df, shock_df, baseline_days, shock_days),
        temporal_disruption_features(baseline_df, shock_df),
        communication_volume_features(baseline_df, shock_df, baseline_days, shock_days),
        mobility_disruption_features(baseline_df, shock_df),
        international_call_surge_features(baseline_df, shock_df, baseline_days, shock_days),
        balance_of_contacts_shift_features(baseline_df, shock_df),
        recharge_stress_features(baseline_df, shock_df, baseline_days, shock_days),
        mobile_data_features(baseline_df, shock_df, baseline_days, shock_days),
        mobile_money_features(baseline_df, shock_df, baseline_days, shock_days),
        network_concentration_features(baseline_df, shock_df),
        response_rate_features(baseline_df, shock_df),
        routine_collapse_features(baseline_df, shock_df),
    ]
    # Drop empty frames (feature not available in this deployment)
    parts = [p for p in parts if len(p.columns) > 1]

    result = parts[0]
    for part in parts[1:]:
        result = result.merge(part, on='name', how='outer')
    return result
