# Retries

Relayr retries a delivery when the destination returns a 5xx status or the connection fails. It
does not retry on 4xx: a rejected message is a bug in the payload, not a transient failure.

## Configuring retries

```python
from relayr import Client, RetryPolicy

client = Client(
    api_key="rk_live_...",
    retry_policy=RetryPolicy(max_attempts=8, backoff="exponential", max_delay_seconds=900),
)
```

- `max_attempts` defaults to 5 and may be set between 1 and 10.
- `backoff` is `exponential` (default) or `fixed`.
- `max_delay_seconds` caps the wait between attempts. Default is 600; the maximum is 3600.
- `retry_on_timeout` defaults to true.

## Backoff schedule

With exponential backoff, attempt *n* waits `2^n` seconds plus jitter, capped at
`max_delay_seconds`. The total time to exhaust 5 attempts is about 31 seconds.

## Giving up

After `max_attempts`, the receipt's status becomes `failed` and a `delivery.failed` event fires.
Relayr does not move failed messages to a dead-letter queue automatically; see
[Dead letters](dead-letters.md).
