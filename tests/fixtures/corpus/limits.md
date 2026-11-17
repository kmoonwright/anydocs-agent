# Limits

These limits apply to v3 and are enforced per API key.

- 500 sends per second, burst to 2000.
- Payloads up to 256 KB.
- `max_attempts` between 1 and 10.
- `max_delay_seconds` up to 3600.
- Idempotency keys up to 255 characters, retained 24 hours.
- Receipts retained 90 days; failure-event payloads retained 7 days.

Exceeding the send rate returns HTTP 429 with a `Retry-After` header. The SDK honors it
automatically and does not count a 429 against `max_attempts`.
