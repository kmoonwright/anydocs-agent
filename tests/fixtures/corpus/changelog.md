# Changelog

## 2026-06-01 (v3)

- `idempotency_key` is now required on `send`.
- `max_attempts` default changed from 3 to 5.
- Receipt retention extended to 90 days.
- Removed `RetryPolicy(backoff="linear")`.

## 2026-02-15

- Added `client.replay(receipt_id)`.
- Failure-event payload retention set to 7 days.

## 2025-11-01 (v2)

- Added delivery receipts.
- Added `RetryPolicy`.
