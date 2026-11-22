"""Nonsecret defaults in JSON; deployment credentials in the environment."""
import json
import os
from pathlib import Path
from src.api_integration import DataGovInAPI, GemGovAPI, ProcurementDataAggregator

ROOT = Path(__file__).resolve().parents[1]


def load_settings():
    return json.loads((ROOT / 'config/settings.json').read_text())


def configured_aggregator(settings, secrets=None):
    secrets = secrets or {}
    def setting(name, default=''):
        return os.environ.get(name) or secrets.get(name) or default
    endpoints = settings.get('api_endpoints', {})
    base_url = setting('GEM_API_BASE_URL', endpoints.get('gem', ''))
    gem = GemGovAPI(setting('GEM_API_KEY'), base_url) if base_url else None
    key = setting('DATA_GOV_API_KEY')
    resource = setting('DATA_GOV_RESOURCE_ID', endpoints.get('datagov_resource_id', ''))
    ogd = DataGovInAPI(key, resource, settings.get('datagov_field_map')) if key and resource else None
    return ProcurementDataAggregator(gem, ogd)
