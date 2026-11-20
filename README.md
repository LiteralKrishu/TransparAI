# TransparAI

TransparAI is a Streamlit dashboard for procurement data. It provides contract summaries, vendor concentration metrics, statistical review flags, processing-time trends, financial scenarios, and data exports.

**Status:** CSV and sample data are the enabled sources. Live API mode is disabled by default, as requested. Its adapters need an authorized source and deployment secrets before use. Hosted load, mobile layout, accessibility, and Windows execution still need verification. Statistical flags do not establish fraud, collusion, discrimination, or legal wrongdoing.

## Run locally

Use Python **3.13.14**. Both `app.py` and `dashboard.py` open the same application. `app.py` is the supported entrypoint.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open [the dashboard](http://localhost:8501). On Windows, run `run_dashboard.ps1` or `run_dashboard.bat` from an installation with Python 3.13. The scripts stop if setup fails.

The app starts with generated sample data. The source and last successful load time remain visible. Each browser session keeps its own active dataset. Reloading the browser resets the session.

## Use your data

Select **CSV File**, choose a file, and click **Load Data**. The uploader stays visible. Invalid uploads retain the previous dataset and show an error.

Limits: 10 MB per CSV and 10,000 records. Required columns:

| Column | Meaning |
| --- | --- |
| `contract_id` | Unique contract identifier |
| `vendor_name` | Nonempty vendor name |
| `contract_value` | Finite, nonnegative value in INR |
| `post_date` | Publication date |
| `award_date` | Award date on or after publication |

Dates use ISO 8601 where possible. Time zone offsets are normalized to UTC. Processing time is calculated from the dates. Invalid timelines are rejected. Exact duplicates are removed; conflicting contract IDs are rejected. Optional `category`, `ministry`, and `location` fields default to Unspecified.

Bid comparisons additionally need `tender_id`, `item_id`, `bid_price`, `vendor_name`, `quantity`, and `price_basis`. Include currency and tax/unit assumptions in `price_basis`. Only bids with the same tender, item, quantity, and price basis are compared. Resolve multiple revisions from one vendor before analysis. The app retains at most 100 example pairs and reports the complete count.

## Live sources

There is no built-in public GeM URL. The URL in the original prototype returned 404 during the audit. No unverified procurement dataset is bundled as a live source.

Use deployment environment variables or `.streamlit/secrets.toml`. Never commit real credentials. `.env.example` lists the names; the app does not automatically load `.env`.

| Variable | Purpose |
| --- | --- |
| `GEM_API_BASE_URL` | Authorized HTTPS service base URL, without `/contracts` |
| `GEM_API_KEY` | Optional bearer credential required by that service |
| `DATA_GOV_API_KEY` | API key for Data.gov.in |
| `DATA_GOV_RESOURCE_ID` | Authorized procurement resource identifier |

For Data.gov.in, choose an actual procurement **award** resource with the required fields. A tender closing date is not an award date. The [official OGD help page](https://data.gov.in/help) explains API access and key generation. The GeM adapter expects a compatible `/contracts` endpoint; operator verification of the provider contract is required.

`config/settings.json` stores nonsecret defaults, analysis settings, upload limits, and optional source field mappings. Environment variables override endpoint values. For a resource with different field names, set `datagov_field_map` as canonical name to source field name, for example:

```json
{"contract_id": "award_identifier", "contract_value": "awarded_amount_inr"}
```

After authorized access has been verified, set `live_enabled` to `true` in `config/settings.json`. Select **Government APIs** and click **Load Data**. Sources use bounded pages, request timeouts, a per-process rate limiter, and validation before activation. A failed live fetch never replaces real data with generated records. The previous dataset and its source label stay active. Enable **Refresh live data every 5 minutes** after a successful live load to request periodic updates while that session is open.

## Analysis and settings

- **Home / Data Management:** summaries, cell completeness, previews, CSV and JSON exports.
- **Anomaly Detection:** Z-score, IQR, and Isolation Forest review flags. The expected outlier share affects Isolation Forest output. It is not a fraud probability. No method selected means no result, not LOW risk.
- **Vendor Analysis:** HHI, top-vendor shares, diversity, and a deterministic heatmap of the percentage of contracts flagged by Isolation Forest. Blank cells have no observations.
- **Collusion Detection:** comparable bid-price and vendor/category variation signals. Contract award values alone cannot establish bid collusion.
- **Financial Analysis:** category/cycle summaries and explicit NPV, ROI, IRR, and payback scenarios. Scenario cash flows are user assumptions, not measured returns. IRR supports conventional nonnegative future cash flows.
- **Efficiency / Executive / Statistics:** processing-time distributions, monthly trends, descriptive summaries, and correlations.
- **Clustering / Forecasting:** DBSCAN pattern exploration (first 2,000 records at most) and linear monthly spending projections. Projections are exploratory, not validated budget forecasts.
- **Settings:** detector sensitivity, expected outlier share, grid lines, and legends. Changes persist across page navigation for the current session.

No bias-detection accuracy or WCAG compliance claim is made. Validate such claims with suitable data and independent checks before release.

## Tests and dependencies

```sh
.venv/bin/python -m pip install --require-hashes -r requirements-dev.txt
.venv/bin/python -m pytest tests -q
```

The suite includes numeric checks, validation, API adapter failure and pagination cases, concurrency checks, deterministic charts, CSV upload, navigation, settings, and every dashboard page. Automated tests use mock API responses and do not require government credentials.

The `requirements*.in` files specify direct dependencies. The `requirements*.txt` files lock transitive versions and package hashes. Refresh them with `uv pip compile --python-version 3.13 --universal --generate-hashes requirements.in -o requirements.txt` and the equivalent command for `requirements-dev.in`. Run the complete suite after any update. CI runs on Linux with Python 3.13.14.

To generate sample files:

```sh
.venv/bin/python src/data_generator.py
```

Both outputs are serialized before either existing file is replaced. Each file is replaced atomically. Sample records are fictional even when vendor names resemble real organizations.

## Deploy

```sh
docker build -t transparai .
docker run --rm -p 8501:8501 --env-file .env transparai
```

The container runs as a non-root user and includes a Streamlit health check. Keep it behind HTTPS. Do not expose deployment credentials in client input, logs, or repository files. No production deployment is performed by this repository.

Before a public release:

1. Verify the selected source, schema, credentials, data rights, and last-successful-fetch label.
2. Test a realistic dataset and concurrent browser sessions on the chosen host.
3. Set CPU, memory, and connection limits at the hosting layer. The API limiter is process-local. A deployment with multiple workers needs a shared upstream quota policy.
4. Monitor health, upstream failures, request latency, resource use, and data freshness. Alert on sustained failures. API exceptions omit URLs that can contain credentials.
5. Keep a known working image and dependency lock for rollback. User datasets are session-local; arrange source backups outside the app when required.
6. Test keyboard navigation, screen readers, contrast, chart alternatives, mobile layouts, and downloads. Adjust resource limits based on observed load.

See [the original audit](PRODUCTION_READINESS_AUDIT.md) and [the remediation record](audit/REMEDIATION.md) for evidence and remaining verification work.

## License

[MIT](LICENSE).
