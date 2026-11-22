"""Configured government data clients and a shared validation boundary."""
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import wraps
import logging
import threading
import time
from urllib.parse import quote, urlparse

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)
MAX_RECORDS = 10_000


class RateLimiter:
    """Thread-safe rolling window within one server process."""
    def __init__(self, calls, period):
        if calls < 1 or period <= 0:
            raise ValueError('Rate limit must be positive')
        self.calls, self.period = calls, period
        self.calls_made, self.lock = deque(), threading.Lock()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            while True:
                with self.lock:
                    now = time.monotonic()
                    while self.calls_made and self.calls_made[0] <= now - self.period:
                        self.calls_made.popleft()
                    if len(self.calls_made) < self.calls:
                        self.calls_made.append(now)
                        break
                    delay = self.period - (now - self.calls_made[0])
                time.sleep(max(delay, 0))
            return func(*args, **kwargs)
        return wrapper


class _JSONClient:
    _cooldowns = {}
    _cooldown_lock = threading.Lock()

    def __init__(self):
        self.session, self.last_error = requests.Session(), None

    def _get(self, url, *, params=None, headers=None):
        self.last_error = None
        host = urlparse(url).hostname
        with self._cooldown_lock:
            if time.monotonic() < self._cooldowns.get(host, 0):
                self.last_error = 'Data service is in its requested retry cooldown'
                return None
        try:
            response = self.session.get(url, params=params, headers=headers,
                                        timeout=(5, 20), allow_redirects=False)
            if response.status_code != 200:
                self.last_error = f'HTTP {response.status_code} from configured data service'
                if response.status_code == 429:
                    self.last_error += '; retry after the provider limit'
                    retry = response.headers.get('Retry-After', '60')
                    try:
                        delay = float(retry)
                    except ValueError:
                        try:
                            delay = (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds()
                        except (TypeError, ValueError, OverflowError):
                            delay = 60
                    if not np.isfinite(delay):
                        delay = 60
                    with self._cooldown_lock:
                        self._cooldowns[host] = time.monotonic() + max(delay, 1)
                logger.warning(self.last_error)
                return None
            payload = response.json()
            if not isinstance(payload, dict) or payload.get('error'):
                self.last_error = 'Data service returned an error or unsupported response'
                return None
            return payload
        except (requests.RequestException, ValueError):
            # Exception text can contain API keys in query strings.
            self.last_error = 'Could not read the data service; check connectivity and access'
            logger.warning(self.last_error)
            return None


class GemGovAPI(_JSONClient):
    """Adapter for an operator-provided, authorized GeM-compatible API."""
    def __init__(self, api_key=None, base_url=None):
        super().__init__()
        self.base_url = (base_url or '').rstrip('/')
        parsed = urlparse(self.base_url)
        if self.base_url and (parsed.scheme != 'https' or not parsed.hostname or
                              parsed.username or parsed.query or parsed.fragment):
            raise ValueError('GeM base URL must use HTTPS without credentials or query parameters')
        self.headers = {'Accept': 'application/json', 'User-Agent': 'TransparAI/1.0'}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'

    def _request(self, path, params=None):
        if not self.base_url:
            self.last_error = 'No authorized GeM base URL configured'
            return None
        return self._get(f'{self.base_url}/{path}', params=params, headers=self.headers)

    @RateLimiter(calls=10, period=60)
    def get_contracts(self, filters=None, page=1, limit=100):
        return self._request('contracts', {**(filters or {}), 'page': page, 'limit': limit})

    @RateLimiter(calls=10, period=60)
    def get_contract_details(self, contract_id):
        return self._request(f'contracts/{quote(str(contract_id), safe="")}')

    @RateLimiter(calls=5, period=60)
    def search_vendors(self, vendor_name):
        return self._request('vendors/search', {'name': vendor_name})

    @RateLimiter(calls=5, period=60)
    def get_vendor_details(self, vendor_id):
        return self._request(f'vendors/{quote(str(vendor_id), safe="")}')


class DataGovInAPI(_JSONClient):
    BASE_URL = 'https://api.data.gov.in/resource'

    def __init__(self, api_key, resource_id=None, field_map=None):
        super().__init__()
        self.api_key, self.resource_id = api_key, resource_id
        self.field_map = field_map or {}

    @RateLimiter(calls=5, period=60)
    def get_resource(self, resource_id, filters=None, limit=100, offset=0):
        if not self.api_key or not resource_id:
            self.last_error = 'Data.gov.in needs an API key and a procurement resource ID'
            return None
        params = {**(filters or {}), 'api-key': self.api_key, 'format': 'json',
                  'limit': limit, 'offset': offset}
        return self._get(f'{self.BASE_URL}/{quote(str(resource_id), safe="")}',
                         params=params, headers={'Accept': 'application/json'})


class DataValidator:
    REQUIRED_FIELDS = ['contract_id', 'vendor_name', 'contract_value', 'post_date', 'award_date']

    @staticmethod
    def clean_contract_data(df):
        """Normalize types without fabricating missing values or valid timelines."""
        out = df.copy()
        for col in ['post_date', 'award_date', 'contract_date']:
            if col in out:
                out[col] = pd.to_datetime(out[col], errors='coerce', utc=True, format='mixed').dt.tz_localize(None)
        for col in ['contract_value', 'estimated_value', 'actual_cost', 'bid_price', 'quantity']:
            if col in out:
                out[col] = pd.to_numeric(out[col], errors='coerce')
        for col in ['contract_id', 'vendor_name', 'tender_id', 'item_id']:
            if col in out:
                out[col] = out[col].astype('string').str.strip().replace('', pd.NA)
        for col in ['category', 'ministry', 'location']:
            out[col] = out[col].fillna('Unspecified').astype(str) if col in out else 'Unspecified'
        if 'post_date' in out and 'award_date' in out:
            days = (out.award_date - out.post_date).dt.total_seconds() / 86400
            out['processing_days'] = days.where(days >= 0)
        return out.drop_duplicates().reset_index(drop=True)

    @staticmethod
    def validate_contract_data(df):
        errors = []
        if df.empty:
            errors.append('Dataset has no records')
        if len(df) > MAX_RECORDS:
            errors.append(f'Dataset exceeds the {MAX_RECORDS:,} record limit')
        for col in DataValidator.REQUIRED_FIELDS:
            if col not in df:
                errors.append(f'Missing required field: {col}')
            elif df[col].isna().any() or df[col].astype('string').str.strip().eq('').any():
                errors.append(f'Missing values in field: {col}')
        if 'contract_value' in df:
            values = pd.to_numeric(df.contract_value, errors='coerce')
            if (~np.isfinite(values)).any() or (values < 0).any():
                errors.append('Contract values must be finite, nonnegative numbers')
        if 'contract_id' in df and df.contract_id.duplicated().any():
            errors.append('Duplicate contract IDs found; resolve conflicting records')
        if 'post_date' in df and 'award_date' in df:
            post = pd.to_datetime(df.post_date, errors='coerce', utc=True, format='mixed')
            award = pd.to_datetime(df.award_date, errors='coerce', utc=True, format='mixed')
            if post.isna().any() or award.isna().any():
                errors.append('Invalid or missing post/award dates')
            if (award < post).any():
                errors.append('Award date before post date')
        return not errors, errors

    @staticmethod
    def prepare_contract_data(df):
        cleaned = DataValidator.clean_contract_data(df)
        valid, errors = DataValidator.validate_contract_data(cleaned)
        if not valid:
            raise ValueError('; '.join(errors))
        return cleaned

    @staticmethod
    def get_data_quality_report(df):
        dates = pd.to_datetime(df.get('post_date', pd.Series(dtype='datetime64[ns]')),
                               errors='coerce', utc=True, format='mixed').dropna()
        return {'total_records': len(df), 'duplicate_records': int(df.duplicated().sum()),
                'unique_vendors': df.vendor_name.nunique() if 'vendor_name' in df else 0,
                'date_range': {'start': dates.min().isoformat(), 'end': dates.max().isoformat()} if len(dates) else None,
                'missing_values': df.isna().sum().to_dict(), 'data_types': df.dtypes.to_dict(),
                'completeness_pct': float(df.notna().to_numpy().mean() * 100) if df.size else 0.0}


class ProcurementDataAggregator:
    """Read configured sources. Never substitute synthetic data for live data."""
    def __init__(self, gem_api=None, datagov_api=None):
        self.gem_api, self.datagov_api = gem_api, datagov_api

    @staticmethod
    def _records(payload):
        if isinstance(payload, dict):
            for key in ['records', 'data', 'contracts', 'results']:
                if isinstance(payload.get(key), list):
                    return payload[key]
        return []

    @staticmethod
    def _normalize(records, source, field_map=None):
        aliases = {'contract_id': ['contract_id', 'id', 'contractId', 'tender_id'],
                   'vendor_name': ['vendor_name', 'vendorName', 'vendor', 'contractor'],
                   'contract_value': ['contract_value', 'contractValue', 'value', 'amount'],
                   'post_date': ['post_date', 'postDate', 'publish_date'],
                   'award_date': ['award_date', 'awardDate'],
                   'ministry': ['ministry', 'department']}
        rows = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError('Data service returned an invalid record')
            row = {key: record.get(key) for key in ['category', 'ministry', 'location',
                'tender_id', 'item_id', 'bid_price', 'quantity', 'price_basis']}
            for col, candidates in aliases.items():
                names = [field_map[col]] if field_map and col in field_map else candidates
                row[col] = next((record[n] for n in names if record.get(n) is not None), None)
            for col, name in (field_map or {}).items():
                row[col] = record.get(name)
            row['source'] = source
            rows.append(row)
        return DataValidator.clean_contract_data(pd.DataFrame(rows)) if rows else pd.DataFrame()

    def _fetch_from_gem(self, max_records):
        records, seen = [], set()
        for page in range(1, (max_records + 99) // 100 + 1):
            payload = self.gem_api.get_contracts(page=page, limit=min(100, max_records-len(records)))
            if payload is None:
                raise ValueError(getattr(self.gem_api, 'last_error', None) or 'GeM request failed')
            batch = self._records(payload)
            if not batch or repr(batch) in seen:
                break
            seen.add(repr(batch))
            records.extend(batch[:max_records-len(records)])
            if len(records) >= max_records:
                break
        return self._normalize(records, 'GeM')

    def _fetch_from_datagov(self, max_records):
        client, records = self.datagov_api, []
        while len(records) < max_records:
            limit = min(1000, max_records-len(records))
            payload = client.get_resource(client.resource_id, limit=limit, offset=len(records))
            if payload is None:
                raise ValueError(client.last_error or 'Data.gov.in request failed')
            batch = self._records(payload)
            records.extend(batch[:limit])
            if len(batch) < limit:
                break
        return self._normalize(records, 'Data.gov.in', client.field_map)

    def fetch_and_aggregate_data(self, start_date=None, end_date=None, max_records=500):
        if not 1 <= max_records <= MAX_RECORDS:
            raise ValueError(f'max_records must be between 1 and {MAX_RECORDS}')
        frames, errors = [], []
        for name, client, fetch in [('GeM', self.gem_api, self._fetch_from_gem),
                                    ('Data.gov.in', self.datagov_api, self._fetch_from_datagov)]:
            if client is None:
                continue
            try:
                frame = fetch(max_records)
                if frame.empty:
                    errors.append(f'{name}: No records returned')
                    continue
                frames.append(DataValidator.prepare_contract_data(frame))
            except ValueError as exc:
                errors.append(f'{name}: {exc}')
        if not frames:
            return pd.DataFrame(), 'No live data', errors or ['No authorized data source configured']
        combined = pd.concat(frames, ignore_index=True)
        compare = combined.drop(columns=['source'], errors='ignore')
        combined = combined.loc[~compare.duplicated()].copy()
        if combined.contract_id.duplicated().any():
            return pd.DataFrame(), 'No live data', errors + ['Conflicting contract IDs across sources']
        if start_date is not None:
            combined = combined[combined.post_date >= pd.to_datetime(start_date, utc=True).tz_localize(None)]
        if end_date is not None:
            combined = combined[combined.post_date <= pd.to_datetime(end_date, utc=True).tz_localize(None)]
        combined = combined.head(max_records).reset_index(drop=True)
        if combined.empty:
            return combined, 'No live data', errors + ['No records match the selected date range']
        return combined, ' + '.join(sorted(combined.source.unique())), errors
