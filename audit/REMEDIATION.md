# Audit remediation record

Date: 21 November 2026 (India time).

The 20 November audit is the baseline record. The changes below address its code defects. They do not certify a hosted production deployment.

| Finding | Change |
| --- | --- |
| F01 | Replaced random heatmap values with reproducible per-vendor/category model flag rates. Labels and matrix dimensions agree. Missing combinations remain blank. |
| F02 | Removed the broken default endpoint and unauthenticated sample resource. Added configured GeM and Data.gov.in adapters, environment/Streamlit secrets, schema mappings, and explicit failed-fetch states. An authorized live source is still required for end-to-end verification. |
| F03 | Added compatible default arguments and regression tests. The page renders without a missing-argument exception. |
| F04 | Made upload persistent. Added size/row limits, normalization, validation, and preservation of the previous dataset on failure. |
| F05 | Removed the old independent-date generator with the duplicate app. Shared ingestion preserves missing/reversed dates and rejects them. Sample award dates follow post dates. |
| F06 | Added finite numeric and nonempty/schema validation before activating data. Small datasets return an empty set of model flags without a crash. |
| F07 | Removed the rate-based HIGH/MEDIUM/LOW verdict and unsupported collusion conclusions. Bid similarity requires comparable tender/item/quantity/price-basis data. The UI explains the limits. |
| F08 | Connected quick-action callbacks to the navigation radio state. |
| F09 | Read defaults from configuration. Persisted session settings and passed them into detectors and charts. |
| F10 | Calculated completeness from cells and handled empty frames. |
| F11 | Fixed JSON serialization and staged both sample outputs before replacing either file. Each replacement is atomic; the two-file pair is not a filesystem transaction. |
| F12 | Made app.py run the same dashboard as dashboard.py. Updated both launch scripts and the README. |
| F13 | Added pagination limits, exact deduplication, conflicting-ID rejection, independent date bounds, and no synthetic live fallback. |
| F14 | Added a lock and monotonic clock to the process-local rolling limiter. Timestamps are recorded after waiting. HTTP failures stop the fetch. A 429 response sets a process-wide provider cooldown from Retry-After. |
| F15 | Preserved Z-score mask alignment. Handled constant values and zero durations. Invalid model inputs are rejected. |
| F16 | Removed the separate basic chart implementation. The shared concentration chart includes Others in its denominator. |
| F17 | Converted vendor chart periods to strings before serialization. |
| F18 | Counted pairs with sorted search windows and retained a bounded number of examples. Enforced dataset/upload limits. |

Financial scenarios, clustering, and monthly projections are now reachable in the UI. Unvalidated bias detection and accessibility claims were removed. Added locked dependencies with hashes, Python runtime selection, Linux CI, a container definition, deployment guidance, and ignored local credentials/artifacts. Removed tracked bytecode.

## Verification

The final suite passed all 67 tests. Measured statement coverage is 84.4% across the application, source modules, and utilities. A temporary local Streamlit server returned HTTP 200 for health and root requests. The dependency audit reported no known vulnerabilities in the 45 applicable locked packages checked on macOS. Full results are recorded in `2026-11-21-verification.json`. The suite checks the original tests plus the audit regressions and app flows. It does not call government services. No repository sample dataset is overwritten by the tests.

The user selected CSV and sample data for the current release. Live API mode is disabled by default. The live service contract, credentials, and resource selection remain external inputs for future activation. Hosted performance, sustained concurrency, accessibility, mobile layouts, Windows scripts, and browser download completion remain release checks. Docker is installed locally, but its daemon was unavailable, so the image was not built during this work.
