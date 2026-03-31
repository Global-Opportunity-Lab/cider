from datetime import date

import numpy as np
import pandas as pd
import pytest

from helpers.derived_features import (
    balance_of_contacts_shift_features,
    communication_volume_features,
    compute_all_derived_features,
    compute_trend_features,
    direction_asymmetry_features,
    international_call_surge_features,
    mobile_data_features,
    mobile_money_features,
    mobility_disruption_features,
    network_concentration_features,
    new_contact_mobilization_features,
    recharge_stress_features,
    response_rate_features,
    routine_collapse_features,
    silence_features,
    temporal_disruption_features,
)


def _make_df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame({'name': ['a', 'b'], **kwargs})


# ---------------------------------------------------------------------------
# direction_asymmetry_features
# ---------------------------------------------------------------------------

@pytest.mark.unit_test
def test_direction_asymmetry_delta():
    baseline = _make_df(
        percent_initiated_interactions_allweek_allday=0.6,
        percent_initiated_conversations_allweek_allday=0.5,
    )
    shock = _make_df(
        percent_initiated_interactions_allweek_allday=0.4,
        percent_initiated_conversations_allweek_allday=0.3,
    )
    result = direction_asymmetry_features(baseline, shock)
    assert 'percent_initiated_interactions_allweek_allday__delta' in result.columns
    assert 'percent_initiated_conversations_allweek_allday__delta' in result.columns
    pd.testing.assert_series_equal(
        result['percent_initiated_interactions_allweek_allday__delta'],
        pd.Series([-0.2, -0.2], name='percent_initiated_interactions_allweek_allday__delta'),
        check_exact=False,
    )


@pytest.mark.unit_test
def test_direction_asymmetry_ignores_missing_cols():
    baseline = _make_df(percent_initiated_interactions_allweek_allday=0.6)
    shock = _make_df(
        percent_initiated_interactions_allweek_allday=0.4,
        percent_initiated_interactions_allweek_day=0.5,
    )
    result = direction_asymmetry_features(baseline, shock)
    assert 'percent_initiated_interactions_allweek_allday__delta' in result.columns
    assert 'percent_initiated_interactions_allweek_day__delta' not in result.columns


# ---------------------------------------------------------------------------
# silence_features
# ---------------------------------------------------------------------------

@pytest.mark.unit_test
def test_silence_normalises_by_window_length():
    baseline = _make_df(active_days_allweek_allday=300.0)
    shock = _make_df(active_days_allweek_allday=10.0)
    result = silence_features(baseline, shock, baseline_days=365, shock_days=30)
    col = 'active_days_allweek_allday__active_rate_drop_churnable'
    assert col in result.columns
    assert abs(result[col].iloc[0] - (300 / 365 - 10 / 30)) < 1e-6


@pytest.mark.unit_test
def test_silence_columns_are_churnable():
    baseline = _make_df(active_days_allweek_allday=100.0)
    shock = _make_df(active_days_allweek_allday=50.0)
    result = silence_features(baseline, shock, baseline_days=90, shock_days=30)
    assert all('_churnable' in c for c in result.columns if c != 'name')


# ---------------------------------------------------------------------------
# new_contact_mobilization_features
# ---------------------------------------------------------------------------

@pytest.mark.unit_test
def test_new_contact_mobilization_ratio():
    baseline = _make_df(number_of_contacts_allweek_allday_call=100.0)
    shock = _make_df(number_of_contacts_allweek_allday_call=20.0)
    result = new_contact_mobilization_features(baseline, shock, baseline_days=365, shock_days=30)
    col = 'number_of_contacts_allweek_allday_call__contact_rate_ratio'
    assert col in result.columns
    expected = (20 / 30) / (100 / 365)
    assert abs(result[col].iloc[0] - expected) < 1e-6


@pytest.mark.unit_test
def test_new_contact_mobilization_zero_baseline_is_nan():
    baseline = _make_df(number_of_contacts_allweek_allday_call=0.0)
    shock = _make_df(number_of_contacts_allweek_allday_call=5.0)
    result = new_contact_mobilization_features(baseline, shock, baseline_days=30, shock_days=30)
    col = 'number_of_contacts_allweek_allday_call__contact_rate_ratio'
    assert result[col].isna().all()


# ---------------------------------------------------------------------------
# temporal_disruption_features
# ---------------------------------------------------------------------------

@pytest.mark.unit_test
def test_temporal_disruption_delta():
    baseline = _make_df(percent_nocturnal_allweek_call=0.1)
    shock = _make_df(percent_nocturnal_allweek_call=0.3)
    result = temporal_disruption_features(baseline, shock)
    col = 'percent_nocturnal_allweek_call__delta'
    assert col in result.columns
    pd.testing.assert_series_equal(
        result[col],
        pd.Series([0.2, 0.2], name=col),
        check_exact=False,
    )


# ---------------------------------------------------------------------------
# communication_volume_features
# ---------------------------------------------------------------------------

@pytest.mark.unit_test
def test_communication_volume_rate_ratio():
    baseline = _make_df(number_of_interactions_alldir_allweek_allday_call=200.0)
    shock = _make_df(number_of_interactions_alldir_allweek_allday_call=60.0)
    result = communication_volume_features(baseline, shock, baseline_days=365, shock_days=30)
    col = 'number_of_interactions_alldir_allweek_allday_call__volume_rate_ratio'
    assert col in result.columns
    expected = (60 / 30) / (200 / 365)
    assert abs(result[col].iloc[0] - expected) < 1e-6


# ---------------------------------------------------------------------------
# mobility_disruption_features
# ---------------------------------------------------------------------------

@pytest.mark.unit_test
def test_mobility_disruption_antenna_and_home():
    baseline = _make_df(number_of_antennas_allweek_allday=10.0, percent_at_home_allweek_allday=0.3)
    shock = _make_df(number_of_antennas_allweek_allday=3.0, percent_at_home_allweek_allday=0.7)
    result = mobility_disruption_features(baseline, shock)
    assert 'number_of_antennas_allweek_allday__delta' in result.columns
    assert 'percent_at_home_allweek_allday__delta' in result.columns
    assert result['number_of_antennas_allweek_allday__delta'].iloc[0] == pytest.approx(-7.0)
    assert result['percent_at_home_allweek_allday__delta'].iloc[0] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# international_call_surge_features
# ---------------------------------------------------------------------------

@pytest.mark.unit_test
def test_international_call_surge_ratio():
    baseline = _make_df(international_all__recipient_id__count=10.0)
    shock = _make_df(international_all__recipient_id__count=6.0)
    result = international_call_surge_features(baseline, shock, baseline_days=365, shock_days=30)
    col = 'international_all__recipient_id__count__intl_rate_ratio'
    assert col in result.columns
    expected = (6 / 30) / (10 / 365)
    assert abs(result[col].iloc[0] - expected) < 1e-6


@pytest.mark.unit_test
def test_international_call_surge_excludes_non_count_cols():
    baseline = _make_df(
        international_all__recipient_id__count=10.0,
        international_all__day__nunique=5.0,
    )
    shock = _make_df(
        international_all__recipient_id__count=6.0,
        international_all__day__nunique=3.0,
    )
    result = international_call_surge_features(baseline, shock, baseline_days=365, shock_days=30)
    assert 'international_all__recipient_id__count__intl_rate_ratio' in result.columns
    # day__nunique ends with __nunique so should also be included
    assert 'international_all__day__nunique__intl_rate_ratio' in result.columns


# ---------------------------------------------------------------------------
# balance_of_contacts_shift_features
# ---------------------------------------------------------------------------

@pytest.mark.unit_test
def test_balance_of_contacts_shift():
    baseline = _make_df(balance_of_contacts_allweek_allday_call_mean=0.6)
    shock = _make_df(balance_of_contacts_allweek_allday_call_mean=0.4)
    result = balance_of_contacts_shift_features(baseline, shock)
    col = 'balance_of_contacts_allweek_allday_call_mean__delta'
    assert col in result.columns
    assert result[col].iloc[0] == pytest.approx(-0.2)


@pytest.mark.unit_test
def test_balance_of_contacts_only_mean_columns():
    baseline = _make_df(
        balance_of_contacts_allweek_allday_call_mean=0.6,
        balance_of_contacts_allweek_allday_call_std=0.1,
    )
    shock = _make_df(
        balance_of_contacts_allweek_allday_call_mean=0.4,
        balance_of_contacts_allweek_allday_call_std=0.2,
    )
    result = balance_of_contacts_shift_features(baseline, shock)
    assert 'balance_of_contacts_allweek_allday_call_mean__delta' in result.columns
    assert 'balance_of_contacts_allweek_allday_call_std__delta' not in result.columns


# ---------------------------------------------------------------------------
# recharge_stress_features
# ---------------------------------------------------------------------------

@pytest.mark.unit_test
def test_recharge_stress_mean_delta_and_count_ratio():
    baseline = _make_df(recharges_mean=500.0, recharges_count=4.0)
    shock = _make_df(recharges_mean=100.0, recharges_count=10.0)
    result = recharge_stress_features(baseline, shock, baseline_days=90, shock_days=30)
    assert 'recharges_mean__delta' in result.columns
    assert 'recharges_count__rate_ratio' in result.columns
    assert result['recharges_mean__delta'].iloc[0] == pytest.approx(-400.0)
    expected_ratio = (10 / 30) / (4 / 90)
    assert result['recharges_count__rate_ratio'].iloc[0] == pytest.approx(expected_ratio)


# ---------------------------------------------------------------------------
# mobile_data_features
# ---------------------------------------------------------------------------

@pytest.mark.unit_test
def test_mobile_data_rate_ratio():
    baseline = _make_df(mobiledata_total_volume=1000.0, mobiledata_num_days=20.0)
    shock = _make_df(mobiledata_total_volume=300.0, mobiledata_num_days=8.0)
    result = mobile_data_features(baseline, shock, baseline_days=90, shock_days=30)
    assert 'mobiledata_total_volume__rate_ratio' in result.columns
    assert 'mobiledata_num_days__rate_ratio' in result.columns
    expected = (300 / 30) / (1000 / 90)
    assert result['mobiledata_total_volume__rate_ratio'].iloc[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# mobile_money_features
# ---------------------------------------------------------------------------

@pytest.mark.unit_test
def test_mobile_money_outgoing_rate_ratio():
    baseline = _make_df(mobilemoney_outgoing_all_amount_mean=200.0)
    shock = _make_df(mobilemoney_outgoing_all_amount_mean=400.0)
    result = mobile_money_features(baseline, shock, baseline_days=90, shock_days=30)
    col = 'mobilemoney_outgoing_all_amount_mean__rate_ratio'
    assert col in result.columns
    expected = (400 / 30) / (200 / 90)
    assert result[col].iloc[0] == pytest.approx(expected)


@pytest.mark.unit_test
def test_mobile_money_cashout_ratio_delta():
    baseline = _make_df(
        mobilemoney_outgoing_all_amount_mean=100.0,
        mobilemoney_incoming_all_amount_mean=200.0,
    )
    shock = _make_df(
        mobilemoney_outgoing_all_amount_mean=300.0,
        mobilemoney_incoming_all_amount_mean=150.0,
    )
    result = mobile_money_features(baseline, shock, baseline_days=90, shock_days=30)
    col = 'mobilemoney_cashout_ratio__delta'
    assert col in result.columns
    baseline_ratio = 100 / 200
    shock_ratio = 300 / 150
    assert result[col].iloc[0] == pytest.approx(shock_ratio - baseline_ratio)


@pytest.mark.unit_test
def test_mobile_money_zero_incoming_cashout_is_nan():
    baseline = _make_df(
        mobilemoney_outgoing_all_amount_mean=100.0,
        mobilemoney_incoming_all_amount_mean=0.0,
    )
    shock = _make_df(
        mobilemoney_outgoing_all_amount_mean=200.0,
        mobilemoney_incoming_all_amount_mean=0.0,
    )
    result = mobile_money_features(baseline, shock, baseline_days=90, shock_days=30)
    assert result['mobilemoney_cashout_ratio__delta'].isna().all()


# ---------------------------------------------------------------------------
# compute_all_derived_features
# ---------------------------------------------------------------------------

@pytest.mark.unit_test
def test_compute_all_derived_features_joins_cleanly():
    cols = dict(
        percent_initiated_interactions_allweek_allday=0.5,
        percent_nocturnal_allweek_call=0.1,
        active_days_allweek_allday=20.0,
        number_of_contacts_allweek_allday_call=50.0,
        number_of_interactions_alldir_allweek_allday_call=200.0,
        number_of_antennas_allweek_allday=8.0,
        percent_at_home_allweek_allday=0.4,
        international_all__recipient_id__count=5.0,
        balance_of_contacts_allweek_allday_call_mean=0.55,
        recharges_mean=400.0,
        recharges_count=5.0,
        mobiledata_total_volume=500.0,
        mobiledata_num_days=15.0,
        mobilemoney_outgoing_all_amount_mean=150.0,
        mobilemoney_incoming_all_amount_mean=200.0,
    )
    baseline = _make_df(**cols)
    shock = _make_df(**cols)
    result = compute_all_derived_features(baseline, shock, baseline_days=365, shock_days=30)
    assert set(result['name']) == {'a', 'b'}
    assert 'percent_initiated_interactions_allweek_allday__delta' in result.columns
    assert 'active_days_allweek_allday__active_rate_drop_churnable' in result.columns
    assert 'number_of_antennas_allweek_allday__delta' in result.columns
    assert 'mobilemoney_cashout_ratio__delta' in result.columns


@pytest.mark.unit_test
def test_compute_all_derived_features_only_common_names():
    cols = dict(percent_initiated_interactions_allweek_allday=0.5, active_days_allweek_allday=10.0)
    baseline = pd.DataFrame({'name': ['a', 'b'], **cols})
    shock = pd.DataFrame({'name': ['b', 'c'], **cols})
    result = compute_all_derived_features(baseline, shock, baseline_days=90, shock_days=30)
    assert list(result['name']) == ['b']


# ---------------------------------------------------------------------------
# network_concentration_features
# ---------------------------------------------------------------------------

@pytest.mark.unit_test
def test_network_concentration_pareto_delta():
    baseline = _make_df(
        percent_pareto_interactions_allweek_allday_call=0.3,
        percent_pareto_durations_allweek_allday=0.25,
    )
    shock = _make_df(
        percent_pareto_interactions_allweek_allday_call=0.6,
        percent_pareto_durations_allweek_allday=0.5,
    )
    result = network_concentration_features(baseline, shock)
    assert 'percent_pareto_interactions_allweek_allday_call__delta' in result.columns
    assert 'percent_pareto_durations_allweek_allday__delta' in result.columns
    assert result['percent_pareto_interactions_allweek_allday_call__delta'].iloc[0] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# response_rate_features
# ---------------------------------------------------------------------------

@pytest.mark.unit_test
def test_response_rate_delta():
    baseline = _make_df(response_rate_text_allweek_allday=0.7)
    shock = _make_df(response_rate_text_allweek_allday=0.2)
    result = response_rate_features(baseline, shock)
    col = 'response_rate_text_allweek_allday__delta'
    assert col in result.columns
    assert result[col].iloc[0] == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# routine_collapse_features
# ---------------------------------------------------------------------------

@pytest.mark.unit_test
def test_routine_collapse_detects_convergence():
    # Baseline: strong weekday/weekend difference. Shock: no difference.
    baseline = _make_df(active_days_weekday_allday=5.0, active_days_weekend_allday=1.0)
    shock = _make_df(active_days_weekday_allday=3.0, active_days_weekend_allday=3.0)
    result = routine_collapse_features(baseline, shock)
    col = 'active_days_allday_routine_collapse'
    assert col in result.columns
    # baseline strength = |5 - 1| = 4; shock strength = |3 - 3| = 0; delta = -4
    assert result[col].iloc[0] == pytest.approx(-4.0)


@pytest.mark.unit_test
def test_routine_collapse_no_pairs_returns_empty():
    # No _weekday_ columns → no pairs → only 'name' column returned
    baseline = _make_df(active_days_allweek_allday=5.0)
    shock = _make_df(active_days_allweek_allday=3.0)
    result = routine_collapse_features(baseline, shock)
    assert set(result.columns) <= {'name'}


@pytest.mark.unit_test
def test_routine_collapse_column_naming():
    baseline = _make_df(
        percent_nocturnal_weekday_call=0.1,
        percent_nocturnal_weekend_call=0.3,
    )
    shock = _make_df(
        percent_nocturnal_weekday_call=0.2,
        percent_nocturnal_weekend_call=0.2,
    )
    result = routine_collapse_features(baseline, shock)
    # _weekday_ replaced with _ → percent_nocturnal_call_routine_collapse
    assert 'percent_nocturnal_call_routine_collapse' in result.columns


# ---------------------------------------------------------------------------
# compute_trend_features
# ---------------------------------------------------------------------------

@pytest.mark.unit_test
def test_trend_slope_sign_and_magnitude():
    dates = [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)]
    dfs = [
        pd.DataFrame({'name': ['a'], 'recharges_mean': [500.0]}),
        pd.DataFrame({'name': ['a'], 'recharges_mean': [400.0]}),
        pd.DataFrame({'name': ['a'], 'recharges_mean': [300.0]}),
    ]
    result = compute_trend_features(dfs, dates)
    slope = result.loc[result['name'] == 'a', 'recharges_mean__slope'].iloc[0]
    assert slope < 0  # declining


@pytest.mark.unit_test
def test_trend_sign_consistency_perfect_monotone():
    dates = [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1), date(2024, 4, 1)]
    dfs = [
        pd.DataFrame({'name': ['a'], 'x': [float(v)]})
        for v in [10, 8, 6, 4]
    ]
    result = compute_trend_features(dfs, dates)
    assert result.loc[result['name'] == 'a', 'x__sign_consistency'].iloc[0] == pytest.approx(1.0)


@pytest.mark.unit_test
def test_trend_sign_consistency_non_monotone():
    dates = [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1), date(2024, 4, 1)]
    # Down, up, down — slope is negative but one pair goes up
    dfs = [
        pd.DataFrame({'name': ['a'], 'x': [float(v)]})
        for v in [10, 6, 9, 4]
    ]
    result = compute_trend_features(dfs, dates)
    sc = result.loc[result['name'] == 'a', 'x__sign_consistency'].iloc[0]
    assert 0.0 < sc < 1.0


@pytest.mark.unit_test
def test_trend_below_min_windows_returns_nan():
    dates = [date(2024, 1, 1), date(2024, 2, 1)]
    dfs = [
        pd.DataFrame({'name': ['a'], 'x': [10.0]}),
        pd.DataFrame({'name': ['a'], 'x': [8.0]}),
    ]
    result = compute_trend_features(dfs, dates, min_windows=3)
    assert result.loc[result['name'] == 'a', 'x__slope'].isna().all()


@pytest.mark.unit_test
def test_trend_handles_person_missing_from_some_windows():
    dates = [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)]
    dfs = [
        pd.DataFrame({'name': ['a', 'b'], 'x': [10.0, 20.0]}),
        pd.DataFrame({'name': ['a'], 'x': [8.0]}),          # b missing
        pd.DataFrame({'name': ['a', 'b'], 'x': [6.0, 18.0]}),
    ]
    result = compute_trend_features(dfs, dates, min_windows=3)
    # 'a' appears in all 3, 'b' only in 2 → b gets NaN with min_windows=3
    assert not result.loc[result['name'] == 'a', 'x__slope'].isna().any()
    assert result.loc[result['name'] == 'b', 'x__slope'].isna().all()


@pytest.mark.unit_test
def test_trend_requires_at_least_two_windows():
    with pytest.raises(ValueError, match='At least 2 windows'):
        compute_trend_features(
            [pd.DataFrame({'name': ['a'], 'x': [1.0]})],
            [date(2024, 1, 1)],
        )


@pytest.mark.unit_test
def test_trend_mismatched_lengths_raises():
    with pytest.raises(ValueError, match='same length'):
        compute_trend_features(
            [pd.DataFrame({'name': ['a'], 'x': [1.0]}),
             pd.DataFrame({'name': ['a'], 'x': [2.0]})],
            [date(2024, 1, 1)],
        )


# ---------------------------------------------------------------------------
# compute_all_derived_features (updated)
# ---------------------------------------------------------------------------

@pytest.mark.unit_test
def test_compute_all_gracefully_skips_missing_feature_sets():
    # Only CDR columns present — mobile money / recharges / mobiledata absent
    cols = dict(
        active_days_allweek_allday=20.0,
        percent_initiated_interactions_allweek_allday=0.5,
    )
    baseline = _make_df(**cols)
    shock = _make_df(**cols)
    result = compute_all_derived_features(baseline, shock, baseline_days=90, shock_days=30)
    assert set(result['name']) == {'a', 'b'}
    assert 'active_days_allweek_allday__active_rate_drop_churnable' in result.columns
    # No mobile money columns should appear
    assert not any('mobilemoney' in c for c in result.columns)
