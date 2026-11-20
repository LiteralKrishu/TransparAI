"""Regression checks for the production readiness audit findings."""
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.analytics import AnomalyDetection, CollusionDetection, EfficiencyAnalysis
from src.api_integration import (DataValidator, DataGovInAPI, GemGovAPI,
                                 ProcurementDataAggregator, RateLimiter)
from src.data_generator import generate_sample_data, save_sample_data
from src.financial_analysis import FinancialValuation
from src.visualizations import RiskVisualization, TimeSeriesVisualization


@pytest.fixture
def records():
    return [{'contract_id':f'C{i}', 'vendor_name':f'Vendor {i%2}', 'contract_value':100+i,
             'post_date':'2025-01-01', 'award_date':'2025-01-10'} for i in range(5)]


@pytest.mark.parametrize('change', [
    {'contract_value':float('inf')}, {'contract_value':float('nan')},
    {'contract_value':-1}, {'contract_value':'bad'}, {'vendor_name':''},
    {'post_date':None}, {'award_date':'2024-01-01'}, {'award_date':'bad'}])
def test_invalid_data_is_rejected(records, change):
    records[0].update(change)
    with pytest.raises(ValueError):
        DataValidator.prepare_contract_data(pd.DataFrame(records))


def test_normalize_dates_numbers_and_optional_fields(records):
    records[0]['contract_value'] = '100'
    frame = DataValidator.prepare_contract_data(pd.DataFrame(records))
    assert frame.processing_days.tolist() == [9]*5
    assert frame.contract_value.sum() == 510
    assert frame.category.eq('Unspecified').all()
    assert DataValidator.get_data_quality_report(pd.DataFrame(records))['date_range'] is not None


def test_bad_dates_are_not_repaired(records):
    records[0]['award_date'] = '2024-01-01'
    records[1]['award_date'] = None
    cleaned = DataValidator.clean_contract_data(pd.DataFrame(records))
    assert cleaned.processing_days.iloc[:2].isna().all()
    assert not DataValidator.validate_contract_data(cleaned)[0]


def test_empty_and_conflicting_ids(records):
    assert not DataValidator.validate_contract_data(pd.DataFrame(columns=DataValidator.REQUIRED_FIELDS))[0]
    frame = pd.DataFrame(records + [{**records[0], 'contract_value':200}])
    with pytest.raises(ValueError, match='Duplicate'):
        DataValidator.prepare_contract_data(frame)
    assert len(DataValidator.prepare_contract_data(pd.DataFrame(records + [records[0]]))) == 5


def test_completeness_is_cell_based():
    frame = pd.DataFrame({'a':[1,None], 'b':[None,2], 'c':[3,4]})
    assert DataValidator.get_data_quality_report(frame)['completeness_pct'] == pytest.approx(100*4/6)
    assert DataValidator.get_data_quality_report(frame.iloc[:0])['completeness_pct'] == 0


def test_sample_export_pair(tmp_path):
    save_sample_data(tmp_path)
    csv = pd.read_csv(tmp_path/'sample_procurement_data.csv')
    encoded = json.loads((tmp_path/'sample_procurement_data.json').read_text())
    assert len(csv) == len(encoded) == 500
    assert (pd.to_datetime(csv.award_date) >= pd.to_datetime(csv.post_date)).all()
    assert not list(tmp_path.glob('tmp*'))
    assert DataValidator.prepare_contract_data(csv).processing_days.notna().all()


def test_sample_generation_does_not_reset_global_random_state():
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    generate_sample_data(5)
    assert np.random.random() == expected


def test_masks_and_zero_duration():
    flags = AnomalyDetection.zscore_detection(pd.Series([1.,2.,np.nan,100.]), threshold=1)
    assert flags.tolist() == [False, False, False, True]
    assert not AnomalyDetection.zscore_detection(pd.Series([2.,2.])).any()
    df = generate_sample_data(10).assign(processing_days=0)
    assert EfficiencyAnalysis.efficiency_score(df).between(0,1).all()
    assert not AnomalyDetection.isolation_forest_detection(df.iloc[:1], ['contract_value']).any()
    with pytest.raises(ValueError, match='finite'):
        AnomalyDetection.isolation_forest_detection(df.assign(contract_value=np.inf), ['contract_value'])


def test_risk_heatmap_is_reproducible_and_aligned():
    df = generate_sample_data(100)
    flags = np.arange(len(df)) % 2 == 0
    first = RiskVisualization.create_risk_heatmap(df, {'review_flags':flags}).data[0]
    second = RiskVisualization.create_risk_heatmap(df, {'review_flags':flags}).data[0]
    np.testing.assert_equal(first.z, second.z)
    assert first.z.shape == (len(first.y), len(first.x))
    expected = df.assign(flag=flags).groupby(['category','vendor_name']).flag.mean()*100
    for i, category in enumerate(first.y):
        for j, vendor in enumerate(first.x):
            if (category,vendor) in expected.index:
                assert first.z[i,j] == pytest.approx(expected.loc[(category,vendor)])
            else:
                assert np.isnan(first.z[i,j])


def test_vendor_chart_serializes():
    assert TimeSeriesVisualization.create_vendor_performance_timeseries(generate_sample_data(10)).to_json()


def test_unrelated_awards_are_not_competing_bids():
    df = generate_sample_data(10).assign(contract_value=100)
    result = CollusionDetection.detect_bid_clustering(df)
    assert result['available'] is False
    assert result['similar_bids_count'] == 0
    assert not result['suspicious']


def test_bid_pairs_are_bounded_and_grouped():
    n = 1000
    df = pd.DataFrame({'tender_id':['T']*n, 'item_id':['I']*n,
                       'vendor_name':[str(i) for i in range(n)], 'bid_price':[100.]*n,
                       'quantity':[1]*n, 'price_basis':['INR total']*n})
    result = CollusionDetection.detect_bid_clustering(df, max_pairs=20)
    assert result['similar_bids_count'] == n*(n-1)//2
    assert len(result['similar_bids']) == 20
    assert result['truncated'] and not result['suspicious']
    df['item_id'] = df.vendor_name
    assert CollusionDetection.detect_bid_clustering(df)['similar_bids_count'] == 0


def test_rate_limit_records_post_wait_time():
    clock = [0.]
    @RateLimiter(1,60)
    def call(): return clock[0]
    def sleep(delay): clock[0] += delay
    with patch('src.api_integration.time.monotonic', side_effect=lambda:clock[0]), \
         patch('src.api_integration.time.sleep', side_effect=sleep):
        assert [call(),call(),call()] == [0,60,120]


def test_rate_limiter_concurrent_calls():
    import time
    @RateLimiter(1, .02)
    def call(): return time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        calls = sorted(pool.map(lambda _:call(), range(6)))
    assert all(after-before >= .019 for before, after in zip(calls, calls[1:]))


def test_retry_after_is_shared_across_clients():
    first = DataGovInAPI('secret', 'resource')
    second = DataGovInAPI('secret', 'resource')
    response = Mock(status_code=429, headers={'Retry-After':'120'})
    with patch.dict(first._cooldowns, {}, clear=True), \
         patch('src.api_integration.time.monotonic', return_value=100), \
         patch.object(first.session, 'get', return_value=response) as get_first, \
         patch.object(second.session, 'get') as get_second:
        assert first._get('https://api.data.gov.in/resource/test') is None
        assert second._get('https://api.data.gov.in/resource/test') is None
        assert get_first.call_count == 1 and get_second.call_count == 0


def test_api_limit_filter_and_no_sample_fallback(records):
    gem = Mock()
    gem.get_contracts.return_value = {'records':records}
    agg = ProcurementDataAggregator(gem_api=gem)
    df, source, errors = agg.fetch_and_aggregate_data(max_records=3)
    assert len(df) == 3 and source == 'GeM' and not errors
    df, _, _ = agg.fetch_and_aggregate_data(end_date='1900-01-01', max_records=3)
    assert df.empty
    df, _, _ = agg.fetch_and_aggregate_data(start_date='2026-01-01', max_records=3)
    assert df.empty
    gem.get_contracts.return_value = None
    gem.last_error = 'HTTP 404'
    df, source, errors = agg.fetch_and_aggregate_data()
    assert df.empty and source == 'No live data' and errors


def test_ogd_adapter_paginates_and_maps_fields():
    client = DataGovInAPI('secret', 'authorized-resource', {'contract_id':'identifier'})
    rows = [{'identifier':str(i),'vendor_name':'A','contract_value':10,
             'post_date':'2025-01-01','award_date':'2025-01-02'} for i in range(1001)]
    with patch.object(client, 'get_resource', side_effect=[{'records':rows[:1000]}, {'records':rows[1000:]}]) as get:
        df, source, errors = ProcurementDataAggregator(datagov_api=client).fetch_and_aggregate_data(max_records=1001)
    assert len(df) == 1001 and source == 'Data.gov.in' and not errors
    assert get.call_args_list[1].kwargs['offset'] == 1000


def test_http_error_does_not_log_secrets(caplog):
    client = DataGovInAPI('top-secret', 'resource')
    with patch.object(client.session, 'get', side_effect=__import__('requests').ConnectionError('url?api-key=top-secret')):
        assert client.get_resource('resource') is None
    assert 'top-secret' not in caplog.text
    assert client.last_error


def test_no_fabricated_gem_endpoint():
    client = GemGovAPI()
    assert client.get_contracts() is None
    assert 'configured' in client.last_error


def test_financial_known_values_and_irr_error():
    assert FinancialValuation.calculate_irr([110], 100) == pytest.approx(.1)
    assert FinancialValuation.calculate_payback_period([100,100],150) == 1.5
    assert FinancialValuation.calculate_npv([110],.1,100) == pytest.approx(0)
    with pytest.raises(ValueError):
        FinancialValuation.calculate_irr([0],100)
