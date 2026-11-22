# TransparAI feature and production readiness audit

Date: 20 November 2026 (India time)  
Reviewed commit: `0e9aff123af08d56589828c4fd8f1163179686e3`  
Baseline decision: **Do not release as a production procurement analysis service.**

Follow-up: code fixes and verification are recorded in [audit/REMEDIATION.md](audit/REMEDIATION.md). The findings below describe the reviewed baseline commit, before those fixes.

The application works as a sample-data demonstration. Several visible controls do not work, one analysis page crashes, and the risk heatmap displays random scores. Neither configured live data request succeeded during this audit. These issues affect data integrity and core user tasks.

This was an audit. Application code, tests, configuration, and sample datasets were not changed. The audit added this report and its evidence file. The pre-existing untracked `.DS_Store` was left in place.

## Scope and evidence

The README and visible dashboard controls define the intended behavior for this review. No separate product specification or deployment target was present. Both `app.py` and `dashboard.py`, all source modules, utilities, tests, launch scripts, configuration, and sample data were inspected.

The requested auto-skill-router was used to select the code review checklist. The requested audit-skills guide was read. It targets AI skill bundles, which this repository does not contain. Its static threat checks informed the security review, but its bundle score would not be a valid application security rating.

Validation used a temporary environment on macOS with Python 3.13.14. Important resolved versions were Streamlit 1.67.0, pandas 3.1.1, NumPy 2.5.5, scikit-learn 1.9.1, SciPy 1.18.1, and Plotly 7.1.0. These versions satisfy the unbounded requirements file. They do not establish compatibility with every permitted version or operating system.

| Check | Observed result |
| --- | --- |
| Parse all 11 Python files | Passed |
| Existing complete test suite | **29 passed, 2 failed** |
| Existing suite coverage | 38% of measured `src` and `utils` statements; neither dashboard was imported |
| Documented entrypoint, simulated API connection failure | Rendered fallback data without an uncaught exception |
| Advanced dashboard, sample data | 9 of 10 navigation destinations rendered without an uncaught exception |
| Home quick actions | All 3 failed to navigate |
| CSV upload control after Load Data | No uploader remained; source stayed Sample Data |
| Configured GeM endpoint | HTTP 404, HTML response |
| Configured Data.gov.in endpoint | HTTP 400, JSON: `Authorization field missing` |
| Dependency vulnerability scan | No known vulnerabilities reported for 78 installed packages, including audit tools |
| Deterministic vendor and finance examples | HHI, concentration ratio, diversity, NPV, IRR, payback, and simple ROI matched expected values |

Detailed results, test failures, package versions, and API responses are in `audit/2026-11-20-results.json`. Page rendering is not proof that every displayed metric is correct.

## Intended feature status

| Intended feature | Status | Evidence or gap |
| --- | --- | --- |
| Contract counts, total value, average value, vendor counts | Working with valid sample data | Both entrypoints render these metrics |
| Isolation Forest detection and interactive scatter plot | Partial | Valid sample data renders; malformed numeric values can crash the basic dashboard; risk interpretation is not calibrated |
| Z-score and IQR detection | Partial | Available in advanced dashboard; Z-score output loses alignment when values are missing |
| Vendor distribution, HHI, CR-4, diversity | Partial | Known numerical examples pass; basic pie excludes the rest of the market; advanced risk heatmap is random |
| Processing-time and monthly trend analysis | Partial | Advanced sample pages render; basic sample data has negative processing times |
| GeM live procurement data | Blocked | Configured endpoint returned 404 |
| Supplementary Data.gov.in data | Incomplete | Dedicated adapter returns `None`; fallback request has no API key |
| CSV import | Broken | Uploader is removed by the immediate rerun; raw CSV dates also crash the quality report |
| Collusion detection | Broken | Required function argument is missing from both dashboard calls and two tests |
| Financial summary and category/cycle cost analysis | Working with valid sample data | Advanced financial page renders |
| NPV, IRR, payback, ROI, budget forecasting | Library-only | Functions exist; the Financial Analysis page does not expose these calculations |
| Executive dashboard and statistical summary | Working with valid sample data | Both advanced pages render; no independent statistical validation of all outputs |
| Home quick actions | Broken | State changes do not control the navigation radio |
| Sensitivity, grid, and legend settings | Inactive | Controls have no connection to the analysis or chart settings |
| CSV and JSON downloads | Partial | Serialization and download widgets work for valid sample data; browser download completion was not tested |
| Sample-data regeneration command | Broken | JSON conversion passes the same argument twice, after writing the CSV |
| DBSCAN and additional vendor charts | Library-only / partial | No UI route; vendor time-series chart fails JSON serialization |
| Bias detection | Not demonstrated | No defined fairness criteria, suitable labels, or validation evidence was found |
| Responsive design and WCAG AA | Unverified | No browser, keyboard, screen-reader, or mobile-layout audit was performed |
| Automatic live refresh | Missing | No timed refresh or scheduled ingestion exists |

## Findings that block release

Priority P1 means that the issue should be fixed before release. P2 means that the issue needs correction before the affected feature is offered for production use.

### F01 — P1: Vendor risk scores are random

Location: `src/visualizations.py:33–45`; called by `dashboard.py:300`.

`create_risk_heatmap` ignores `risk_metrics` and uses `np.random.rand` for scores. Two calls with identical data produce different risk values. The test also found 22 score columns with only 10 vendor labels. A user can interpret an arbitrary value as evidence against a named vendor.

Replace the random matrix with documented, reproducible calculations. Align each score with its vendor and category. Show unavailable data explicitly. Verify stable output for fixed input before restoring this chart.

### F02 — P1: Live government data is not operational

Locations: `app.py:138–141`; `dashboard.py:128–132`; `src/api_integration.py:425–430,464–472`.

The configured GeM contracts URL returned HTTP 404. The additional Data.gov.in request returned HTTP 400 with an authorization error. The dedicated Data.gov.in fetch method is a stub, and the dashboard supplies no API key to its clients. Configuration lists eProcurement and CPPP sources, but no corresponding fetch implementation was found.

Confirm a supported data source, its access requirements, and its schema. Connect credentials through deployment secrets. Implement and test the selected adapter against an authorized source. Keep sample mode available with a persistent sample-data label. The basic dashboard footer currently names government sources even when it shows generated records.

The official [OGD help page](https://data.gov.in/help) documents API-key generation and access. The observed failures describe these requests on the audit date; they do not prove that all GeM or OGD services are unavailable.

### F03 — P1: Collusion Detection crashes

Locations: `dashboard.py:324,336`; `src/analytics.py:375,408`; `tests/test_analytics.py:202,210`.

Both analysis functions require `value_column`. Their dashboard callers and tests omit it. AppTest reproduced the missing-argument exception on the Collusion Detection page. The same defect accounts for both existing test failures.

Align the function contract and all callers. Add a page test that reaches both sections and checks meaningful expected results.

### F04 — P1: CSV import cannot complete and bypasses normalization

Locations: `dashboard.py:120–125,705–708`; `src/api_integration.py:615–618`.

The uploader is created only while Load Data is true. The immediate `st.rerun()` then removes it. AppTest found zero uploader widgets after the click. The [Streamlit button guide](https://docs.streamlit.io/develop/concepts/design/buttons) describes the temporary nature of button state.

There is a second defect behind this UI failure: `pd.read_csv` stores raw strings without cleaning or validation. Supplying the repository's CSV directly as session data causes the home page to fail with `'str' object has no attribute 'isoformat'`. Missing columns, empty frames, and invalid numeric input also lack a controlled rejection path.

Render the uploader persistently. Normalize dates and numbers before changing the active dataset. Validate required columns and finite values. Reject invalid input with clear errors and retain the previous valid dataset.

### F05 — P1: Invalid dates corrupt processing-time results

Locations: `app.py:110–119,216–217`; `src/api_integration.py:390–415,485–513`.

The basic generator chooses post and award dates independently. The audit found 97 negative durations among 200 records, with a minimum of -341 days. API processing replaces missing duration with zero and applies `abs()` to negative durations. Missing GeM dates can also be replaced with the current time. These operations make invalid or unknown timelines appear valid and efficient.

Generate award dates after post dates. Retain unknown dates as missing. Reject or flag reversed timelines. Exclude invalid durations from efficiency and anomaly calculations with a visible count of excluded records.

### F06 — P1: Invalid live values cause an uncaught dashboard failure

Locations: `app.py:226–245,278–279`; `src/api_integration.py:535–553`.

With a mocked successful API response containing an infinite contract value, anomaly detection reports a scaling error and returns the unchanged frame. The caller then raises `KeyError: 'is_anomaly'`. The shared validator accepts infinite values and empty frames. String contract values cause a `TypeError` instead of a validation result.

Check schema, numeric conversion, finiteness, and minimum usable row count before analysis. Return an explicit failed-analysis result. Render a controlled error state without reading result columns that were never created.

### F07 — P1: Analytical labels exceed the available evidence

Locations: `dashboard.py:235–255,317–343`; `src/analytics.py:388–404,423–433`.

The anomaly percentage comes from an Isolation Forest fitted with a fixed 10% contamination parameter. It is then used to label overall risk. This setting influences the reported outlier rate; it is not an independently measured probability of procurement abuse. Turning Isolation Forest off produces a zero-rate LOW label even if another detector finds outliers.

The bid comparison uses awarded contract values across the whole dataset without tender identity, product quantity, time, or competing bids. A probe with ten equal-value contracts in ten different categories was marked suspicious, with 45 pairs. Similar prices alone do not establish collusion.

Define the intended inference and required data. Present these results as review signals with limitations. Use labeled cases or a justified benchmark to evaluate false positives and missed cases. Remove unsupported risk and collusion conclusions until this validation is complete.

## Other feature defects and production gaps

| ID / priority | Finding and evidence | Required correction |
| --- | --- | --- |
| F08 / P2 | Quick actions write `st.session_state.page`, but routing reads an unkeyed radio. All three remained on Home. `dashboard.py:591–603,686–692,727`. | Use one navigation state value for both the radio and button callbacks. |
| F09 / P2 | Settings values are local and unused. Setting the threshold to 1.0 still produced the default seven Z-score anomalies. `dashboard.py:228,788–792`. No application code reads `config/settings.json`. | Persist settings and pass them to analysis, chart, and API functions. Define which settings file is authoritative. |
| F10 / P2 | Completeness divides missing cells by row count. Ten rows, twelve columns, and six missing cells display 40%, although 95% of cells are present. Three fully missing optional columns display -200%. `dashboard.py:201,533`. | Divide present cells by total cells, or define a separate row-completeness measure. Handle zero rows. |
| F11 / P2 | `save_sample_data` calls `df.to_dict('records', orient='records')`. This raises `TypeError` after the CSV is written. `src/data_generator.py:205–209`. | Specify `orient` once and complete both outputs before replacing the existing pair. |
| F12 / P2 | README and both launch scripts run `app.py`. Advanced pages exist only in `dashboard.py`. | Select and document one supported entrypoint. Connect supported features to it. Remove or clearly label the alternative. |
| F13 / P2 | `max_records=3` returned 5 mocked GeM records. Aggregation does not remove duplicate contracts. Date filters require both bounds and are ignored during sample fallback. `src/api_integration.py:315–344,356–416`. | Apply limits, canonical contract IDs, source-aware deduplication, and each supplied date bound consistently. |
| F14 / P2 | The rate limiter stores the timestamp from before sleep. A virtual-clock probe allowed calls at 0, 60, and 60 seconds with a one-per-minute limit. State is process-local and has no concurrency lock. `src/api_integration.py:37–45`. | Use current monotonic time after waiting. Add concurrency protection appropriate to the deployment. Respect upstream backoff responses. |
| F15 / P2 | Z-score detection drops missing rows and returns a shorter mask. Four input rows produced three flags. All-zero processing days produced 100 missing efficiency scores. `src/analytics.py:34–35,455–470`. | Preserve input alignment and define behavior for missing data, zero ranges, and constant datasets. |
| F16 / P2 | The basic vendor pie contains only the top eight vendors, so its percentages use that subset as the denominator. `app.py:311–319`. | Add the remaining vendors as Others, or explicitly label percentages as share within the displayed subset. |
| F17 / P2 | Vendor time-series output contains pandas Period objects and cannot be serialized to chart JSON. `src/visualizations.py:277–293`. | Convert the period to a supported chart-axis type and verify serialization. |
| F18 / P2 | The collusion pair loop is quadratic and retains every matching pair. No dataset row limit or measured production workload is defined. `src/analytics.py:390–399`. | Define supported dataset size. Limit output and use suitable grouping or indexing before enabling large inputs. |

The NPV, IRR, payback, ROI, and forecasting utilities are not connected to the financial page. This is a feature gap if these library capabilities are intended for users. Their presence alone does not make them dashboard features. Basic known-value finance checks passed, but no comprehensive numerical edge-case audit was completed.

## Security and operational readiness

Static inspection found no hardcoded secret values, remote-script piping, dynamic code execution, privilege escalation, destructive system operations, or obfuscated payloads in the reviewed source. Requests use HTTPS and retain normal certificate verification. Windows launch scripts create an environment, install dependencies, and start Streamlit. They were inspected but not executed on Windows.

These observations are limited to the reviewed files. They are not a penetration-test result. No authentication requirement was inferred for a public, read-only dashboard.

The dependency scan reported zero known vulnerabilities in this temporary environment. The repository has no locked dependency set, so that result does not cover another installation or deployment. There is no CI workflow, deployment manifest, defined Python runtime, or documented production monitoring and recovery procedure. Application warnings are suppressed globally, while API failures rely mainly on print statements or transient UI messages.

The README claims Python 3.8 support, but `pandas>=2.2.0` excludes Python 3.8. The [pandas 2.2.0 package metadata](https://pypi.org/project/pandas/2.2.0/) requires Python 3.9 or later. The versions installed for this audit have higher runtime requirements. Choose a supported Python version and test it with a locked dependency set.

Tracked `__pycache__` files and the absent `.gitignore` are repository hygiene gaps. Flask and Gunicorn are installed but no Flask application or WSGI deployment was found. Confirm the actual serving model before choosing production infrastructure.

## Release gates

1. Fix F01–F07 and verify them with focused regression checks.
2. Restore CSV import, navigation, settings, completeness, and sample-data export.
3. Select one entrypoint and make the README match its actual supported features.
4. Validate a supported live source with authorized credentials, schema checks, persistent provenance, and a visible last-successful-fetch time.
5. Lock the runtime and dependencies. Add automated checks for the complete suite, each supported page, source failure, malformed input, and exports.
6. Test the selected deployment with realistic data size and concurrent sessions. Define timeouts, upstream backoff, resource limits, logs, monitoring, and recovery.
7. Verify mobile layout, keyboard access, chart alternatives, and contrast before retaining the WCAG AA claim.

## Verification limits and reproduction

No production URL or deployment credentials were provided. Hosted behavior, browser downloads, visual accessibility, mobile layout, sustained load, concurrent sessions, and Windows execution remain unverified. Live endpoint checks were read-only, with one request per configured target and no credentials. API adapter edge cases used controlled mock responses.

The sample-data saving command was not run against the repository because it writes existing datasets. Its failing JSON conversion was reproduced on an in-memory frame. AppTest exercised the dashboard scripts without editing them. See the official [AppTest reference](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest) for the test mechanism.

To repeat the existing tests in an installed project environment:

```sh
python -m pytest tests -q
```

The two observed failures are `TestCollusionDetection.test_bid_clustering` and `TestCollusionDetection.test_price_patterns`. The README's narrower `tests/test_dashboard.py` command does not include them.

To reproduce the collusion page failure:

```python
from streamlit.testing.v1 import AppTest
app = AppTest.from_file('dashboard.py').run()
app.radio[0].set_value('Collusion Detection').run()
print([error.message for error in app.exception])
```
