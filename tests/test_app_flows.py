"""Exercise user flows through the supported Streamlit entrypoint."""
from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def app(entry='app.py'):
    return AppTest.from_file(str(ROOT/entry), default_timeout=60).run()


def button(at, label):
    return next(value for value in at.button if value.label == label)


@pytest.mark.parametrize('entry',['app.py','dashboard.py'])
def test_every_page_renders(entry):
    at = app(entry)
    assert not at.exception
    for page in at.radio[0].options:
        at.radio[0].set_value(page).run()
        assert not at.exception, (page, [e.message for e in at.exception])


@pytest.mark.parametrize('label,target',[
    ('🔍 Run Anomaly Detection','Anomaly Detection'),
    ('🏢 Vendor Analysis','Vendor Analysis'),
    ('📊 Executive Dashboard','Executive Dashboard')])
def test_quick_actions_navigate(label,target):
    at = app()
    button(at,label).click().run()
    assert at.radio[0].value == target and not at.exception


def test_csv_upload_valid_then_invalid_keeps_previous_data():
    at = app()
    at.selectbox[0].set_value('CSV File').run()
    button(at,'🔄 Load Data').click().run()
    assert len(at.file_uploader) == 1 and not at.exception
    payload = (ROOT/'data/sample_procurement_data.csv').read_bytes()
    at.file_uploader[0].set_value(('procurement.csv',payload,'text/csv')).run()
    button(at,'🔄 Load Data').click().run()
    assert not at.exception
    assert at.session_state['data_source'] == 'Uploaded CSV'
    previous = at.session_state['data'].copy()
    at.file_uploader[0].set_value(('bad.csv',b'wrong,columns\n1,2','text/csv')).run()
    button(at,'🔄 Load Data').click().run()
    assert at.error and not at.exception
    assert at.session_state['data'].equals(previous)


def test_settings_change_detector_and_survive_navigation():
    at = app()
    at.radio[0].set_value('Settings').run()
    next(v for v in at.slider if v.label == 'Z-score threshold:').set_value(1.0).run()
    next(v for v in at.checkbox if v.label == 'Show grid lines').uncheck().run()
    at.radio[0].set_value('Anomaly Detection').run()
    low = next(v.value for v in at.markdown if 'Z-Score Anomalies' in v.value)
    at.radio[0].set_value('Settings').run()
    assert next(v for v in at.slider if v.label == 'Z-score threshold:').value == 1.0
    assert not next(v for v in at.checkbox if v.label == 'Show grid lines').value
    next(v for v in at.slider if v.label == 'Z-score threshold:').set_value(5.0).run()
    at.radio[0].set_value('Anomaly Detection').run()
    high = next(v.value for v in at.markdown if 'Z-Score Anomalies' in v.value)
    assert low != high and not at.exception
    at.multiselect[0].set_value([]).run()
    assert any('Select a method' in v.value for v in at.info)
    assert not at.exception


def test_live_source_is_disabled_by_default():
    at = app()
    assert at.selectbox[0].options == ['Sample Data', 'CSV File']
    assert at.session_state['data_source'] == 'Sample Data'
    assert not at.session_state['live_loaded']


def test_live_refresh_keeps_last_success_on_failure():
    from datetime import datetime, timedelta, timezone
    from unittest.mock import Mock, patch
    from src.data_generator import generate_sample_data
    live = generate_sample_data(5).assign(source='GeM')
    aggregator = Mock()
    aggregator.fetch_and_aggregate_data.return_value = (live, 'GeM', [])
    from src.configuration import load_settings
    settings = load_settings()
    settings['live_enabled'] = True
    with patch('src.configuration.configured_aggregator', return_value=aggregator), \
         patch('src.configuration.load_settings', return_value=settings):
        at = app()
        at.selectbox[0].set_value('Government APIs').run()
        button(at,'🔄 Load Data').click().run()
        assert at.session_state['live_loaded'] and at.session_state['data_source'] == 'GeM'
        loaded_at = at.session_state['loaded_at']
        previous = at.session_state['data'].copy()
        aggregator.fetch_and_aggregate_data.return_value = (live.iloc[:0], 'No live data', ['HTTP 429'])
        old = (datetime.now(timezone.utc)-timedelta(seconds=600)).isoformat()
        at.session_state['loaded_at'] = old
        next(v for v in at.checkbox if v.label == 'Refresh live data every 5 minutes').check().run()
        assert not at.exception
        assert at.session_state['data'].equals(previous)
        assert at.session_state['loaded_at'] == old
        assert at.session_state['data_source'] == 'GeM'
        assert aggregator.fetch_and_aggregate_data.call_count == 2
