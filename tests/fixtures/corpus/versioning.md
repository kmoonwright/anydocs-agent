# Versioning

Relayr pins behavior to a dated API version sent with every request.

- v3 is current. v2 is in maintenance until 2027-01-01 and receives security fixes only.
- v1 was retired on 2026-01-01.

## Setting a version

```python
client = Client(api_key="rk_live_...", api_version="2026-06-01")
```

If you do not set `api_version`, the client uses the newest version available when the SDK was
released.

## What changed in v3

- `idempotency_key` became required. In v2 it was optional.
- `max_attempts` default rose from 3 to 5.
- Receipt retention went from 30 days to 90 days.
- `RetryPolicy(backoff="linear")` was removed; use `fixed`.
